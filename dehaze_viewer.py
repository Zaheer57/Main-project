import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess

# Function to open the dehazed image
def open_dehazed_image():
    filename = filedialog.askopenfilename(title="Open Dehazed Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if filename:
        image = Image.open(filename)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

# Create a Tkinter window
root = tk.Tk()
root.title("Dehazed Image Viewer")

# Create a button to open the dehazed image
open_button = tk.Button(root, text="Open Dehazed Image", command=open_dehazed_image)
open_button.pack()

# Create a label to display the dehazed image
image_label = tk.Label(root)
image_label.pack()

# Run the Tkinter event loop
root.mainloop()
