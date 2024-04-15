
import streamlit as st
from PIL import Image
import subprocess #handles exceptions,i\o
import os
import glob
import logging
import numpy as np
import io
import PIL.Image as Image
from gf import guided_filter  # Assuming guided_filter is implemented elsewhere

# Custom CSS for background image
def set_background(image_url):
    # Add custom CSS
    st.markdown(
        f"""
        <style>
            .reportview-container {{
                background: url("{image_url}") no-repeat center center fixed;
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def open_image(self, img_path: str) -> None:
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double) / 255.
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)

    def get_dark_channel(self) -> None:
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                self.dark[i, j] = tmp[rmin:rmax + 1, cmin:cmax + 1].min()

    def get_air_light(self) -> None:
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)

    def get_transmission(self) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - self.radius)
                rmax = min(i + self.radius, self.rows - 1)
                cmin = max(0, j - self.radius)
                cmax = min(j + self.radius, self.cols - 1)
                pixel = (self.src[rmin:rmax + 1, cmin:cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1. - self.omega * pixel

    def guided_filter(self) -> None:
        self.gtran = guided_filter(self.src, self.tran, self.r, self.eps)

    def recover(self) -> None:
        self.gtran[self.gtran < self.t0] = self.t0
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        self.dst = (self.src.astype(np.double) - self.Alight) / t + self.Alight
        self.dst *= 255
        self.dst[self.dst > 255] = 255
        self.dst[self.dst < 0] = 0
        self.dst = self.dst.astype(np.uint8)

    def get_output_image_bytes(self) -> bytes:
        image = Image.fromarray(self.dst)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

def detect_objects(input_image_path):
    output_folder = "runs/detect/exp"
    os.makedirs(output_folder, exist_ok=True)

    # Generate the command for object detection
    command = f"python detect.py --source {input_image_path} --weights yolov5s.pt --conf 0.4 --save-txt --save-conf --exist-ok --project runs/detect --name exp"

    try:
        # Execute the command using subprocess
        result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        logging.info("Detection process completed.")

        # Display recent output image
        display_recent_output_image(output_folder)

    except subprocess.CalledProcessError as e:
        # Handle subprocess errors
        logging.error(f"Error executing command: {e}")
        logging.error(f"Command stderr: {e.stderr}")

        # Display error message to the user
        st.error("Error executing object detection. Please check the input image and try again.")

def display_recent_output_image(output_folder):
    files = glob.glob(os.path.join(output_folder, "*.jpg")) + glob.glob(os.path.join(output_folder, "*.png"))
    if files:
        latest_image_path = max(files, key=os.path.getctime)
        latest_image = Image.open(latest_image_path)
        st.image(latest_image, caption="Recent Detected Objects", use_column_width=True)
    else:
        st.write("No output images found in the folder.")

def main():
    st.title("Haze Removal and Object Detection")

    # Set background image
    set_background("C:\\Users\\zahee\\Desktop\\projectui\\projectfinal\\detect\\th.jpg")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        hr = HazeRemoval()
        hr.open_image(uploaded_file)

        st.text("Processing...")

        hr.get_dark_channel()
        hr.get_air_light()
        hr.get_transmission()
        hr.guided_filter()
        hr.recover()

        st.text("Done! Here are the input and dehazed images:")

        # Display input and dehazed images side by side
        col1, col2 = st.columns(2)
        col1.image(hr.src, caption="Input Image", use_column_width=True)
        col2.image(hr.dst, caption="Dehazed Image", use_column_width=True)

        if st.button('Download Dehazed Image'):
            output_bytes = hr.get_output_image_bytes()
            st.download_button(label='Download Dehazed Image', data=output_bytes, file_name='dehazed_image.jpg', mime='image/jpeg')

        detect_button = st.button("Detect Objects")
        if detect_button:
            # Save the dehazed image to a temporary file
            dehazed_image_path = "dehazed_image.jpg"
            Image.fromarray(hr.dst).save(dehazed_image_path)
            detect_objects(dehazed_image_path)  # Pass the path to the dehazed image

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     main()
