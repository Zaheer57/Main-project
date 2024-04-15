import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

def calculate_glcm(image, distance=1, angles=[0], levels=256):
    glcm = np.zeros((levels, levels), dtype=np.uint32)
    height, width = image.shape

    for angle in angles:
        for y in range(height):
            for x in range(width):
                for d in range(1, distance+1):
                    x2, y2 = x, y
                    if angle == 0:
                        x2 += d
                    elif angle == np.pi/4:
                        x2 += d
                        y2 -= d
                    elif angle == np.pi/2:
                        y2 -= d
                    elif angle == 3*np.pi/4:
                        x2 -= d
                        y2 -= d
                    
                    if 0 <= x2 < width and 0 <= y2 < height:
                        glcm[image[y, x], image[y2, x2]] += 1
    glcm_normalized = glcm.astype(float) / np.sum(glcm)

    contrast = np.sum(glcm_normalized * np.square(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])))
    correlation = np.sum((np.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])) - np.sum(glcm_normalized * np.meshgrid(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])))) ** 2)
    energy = np.sum(glcm_normalized ** 2)
    homogeneity = np.sum(glcm_normalized / (1 + np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1]))))
    print(contrast)
    print(correlation)
    print(energy)
    print(homogeneity)
    return glcm

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('L') # convert to grayscale
        img_array = np.array(img)
        glcm = calculate_glcm(img_array)
        print("GLCM calculated successfully.")

# Simple GUI
root = tk.Tk()
root.title("GLCM Calculator")

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

root.mainloop()