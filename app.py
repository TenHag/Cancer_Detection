import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Corrected import

class ImageClassifierUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")

        self.model = load_model('CNN_MODEL.h5')  # Replace with the correct path

        self.canvas = tk.Canvas(self.master, width=300, height=300)
        self.canvas.pack()

        self.label_result = tk.Label(self.master, text="Prediction: ")
        self.label_result.pack()

        self.btn_upload = tk.Button(self.master, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                               filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (300, 300))
            self.display_image(image)

            # Perform prediction using your model
            prediction = self.predict(image)
            self.label_result.config(text=f"Prediction: {prediction}")

    def display_image(self, image):
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Keep a reference to the image to prevent garbage collection

    def predict(self, image):
        image = cv2.resize(image, (64, 64))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        result = self.model.predict(image)
        class_index = np.argmax(result)
        # Replace the labels with your actual class labels
        class_labels = ['No', 'Benign', 'Malignant', 'Normal']
        prediction = class_labels[class_index]
        return prediction


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierUI(root)
    root.mainloop()

