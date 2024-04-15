# import streamlit as st
# from PIL import Image
# import subprocess
# import os
# import glob
# import logging
#
# def detect_objects(input_image_path):
#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#
#     output_folder = "runs/detect/exp"
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Generate the command for object detection
#     command = f"python detect.py --source {input_image_path} --weights yolov5s.pt --conf 0.4 --save-txt --save-conf --exist-ok --project runs/detect --name exp"
#
#     try:
#         # Execute the command using subprocess
#         logger.info(f"Executing command: {command}")
#         result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
#         logger.info(f"Command output: {result.stdout}")
#         logger.info("Detection process completed.")
#
#         # Find the output image
#         output_image_path = find_output_image(output_folder)
#         if output_image_path:
#             logger.info(f"Output image found: {output_image_path}")
#         else:
#             logger.warning("Output image not found.")
#
#         # Display output image
#         display_output_image(output_folder)
#
#     except subprocess.CalledProcessError as e:
#         # Handle subprocess errors
#         logger.error(f"Error executing command: {e}")
#         logger.error(f"Command stderr: {e.stderr}")
#
#         # Display error message to the user
#         st.error("Error executing object detection. Please check the input image and try again.")
#
# def find_output_image(output_folder):
#     files = glob.glob(os.path.join(output_folder, "*.jpg")) + glob.glob(os.path.join(output_folder, "*.png"))
#     if files:
#         return max(files, key=os.path.getctime)
#     return None
#
# def display_output_image(output_folder):
#     output_image_path = find_output_image(output_folder)
#     if output_image_path:
#         output_image = Image.open(output_image_path)
#         st.image(output_image, caption="Objects Detected", use_column_width=True)
#     else:
#         st.write("Output image not found.")
#         st.write("No objects detected.")
#
# # Create Streamlit app
# st.title("Object Detection")
#
# # Create uploads directory if it doesn't exist
# os.makedirs("uploads", exist_ok=True)
#
# # Load Image
# input_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#
# if input_image:
#     input_image_path = os.path.join("uploads", input_image.name)
#     with open(input_image_path, "wb") as f:
#         f.write(input_image.read())
#     st.image(input_image, caption="Uploaded Image", use_column_width=True)
#     detect_button = st.button("Detect Objects")
#
#
#
#




import streamlit as st
from PIL import Image
import subprocess
import os
import glob
import logging

def detect_objects(input_image_path):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    output_folder = "runs/detect/exp"
    os.makedirs(output_folder, exist_ok=True)

    # Generate the command for object detection
    command = f"python detect.py --source {input_image_path} --weights yolov5s.pt --conf 0.4 --save-txt --save-conf --exist-ok --project runs/detect --name exp"

    try:
        # Execute the command using subprocess
        logger.info(f"Executing command: {command}")
        result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        logger.info(f"Command output: {result.stdout}")
        logger.info("Detection process completed.")

        # Find the output image
        output_image_path = find_output_image(output_folder)
        if output_image_path:
            logger.info(f"Output image found: {output_image_path}")
        else:
            logger.warning("Output image not found.")

        # Display output image
        display_output_image(output_image_path)

    except subprocess.CalledProcessError as e:
        # Handle subprocess errors
        logger.error(f"Error executing command: {e}")
        logger.error(f"Command stderr: {e.stderr}")

        # Display error message to the user
        st.error("Error executing object detection. Please check the input image and try again.")

def find_output_image(output_folder):
    files = glob.glob(os.path.join(output_folder, "*.jpg")) + glob.glob(os.path.join(output_folder, "*.png"))
    if files:
        return max(files, key=os.path.getctime)
    return None

def display_output_image(output_image_path):
    if output_image_path:
        output_image = Image.open(output_image_path)
        st.image(output_image, caption="Objects Detected", use_column_width=True)
    else:
        st.write("Output image not found.")
        st.write("No objects detected.")

# Create Streamlit app
st.title("Object Detection")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Load Image
input_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if input_image:
    input_image_path = os.path.join("uploads", input_image.name)
    with open(input_image_path, "wb") as f:
        f.write(input_image.read())
    st.image(input_image, caption="Uploaded Image", use_column_width=True)
    detect_button = st.button("Detect Objects")
    if detect_button:
        detect_objects(input_image_path)
#     if detect_button:
#         detect_objects(input_image_path)
