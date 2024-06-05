import os
import threading
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.ttk import Progressbar
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("50_epoch_Tuberculosis_Model.h5")

# Define the class names
class_names = ["Negative", "Positive"]


# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert to RGB
    img = img.resize((150, 150))  # Resize to the input size of the model
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to predict the class of a single image
def predict_single_image(file_path):
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    confidence = np.max(prediction) * 100
    return confidence


# Function to handle single image prediction
def select_single_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        confidence = predict_single_image(file_path)
        result_text.set(f"Prediction: Chance of Tuberculosis ({confidence:.2f}%)")

        # Display the image in the GUI
        img = Image.open(file_path)
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img


# Function to handle folder prediction and display results in scrollable text
def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        progress_bar["value"] = 0  # Reset progress bar

        # Start a new thread for processing images
        threading.Thread(target=process_folder, args=(folder_path,)).start()


def process_folder(folder_path):
    total_files = len(
        [
            name
            for name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, name))
            and name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    progress_increment = 100 / total_files

    result_area.config(state=tk.NORMAL)  # Enable editing temporarily
    result_area.delete(1.0, tk.END)  # Clear previous results
    result_area.insert("1.0", "Image name: Chance of Tuberculosis\n")
    result_area.tag_add("bold", "1.0", "1.25")
    result_area.tag_configure("bold", font=("Helvetica", 10, "bold"))
    result_area.config(state=tk.DISABLED)  # Disable editing again

    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)
            confidence = predict_single_image(file_path)

            result_area.config(state=tk.NORMAL)  # Enable editing temporarily
            result_area.insert(tk.END, f"{filename}: {confidence:.2f}%\n")
            result_area.config(state=tk.DISABLED)  # Disable editing again

            progress_bar["value"] += progress_increment
            root.update_idletasks()  # Update the GUI

    messagebox.showinfo(
        "Prediction Completed",
        "Prediction of Chest X-ray images in the Test Folder completed.",
    )


# Create the main window
root = tk.Tk()
root.title("Tuberculosis X-ray Classifier")
root.geometry("800x600")  # Set the window size

# Create a label for displaying the result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 12))
result_label.pack(pady=10)

# Create a button for loading and classifying a single image
btn_single = tk.Button(
    root,
    text="Select Single Image",
    command=select_single_image,
    font=("Helvetica", 12),
)
btn_single.pack(pady=10)

# Create a button for loading and classifying images in a folder
btn_folder = tk.Button(
    root, text="Select Folder", command=select_folder, font=("Helvetica", 12)
)
btn_folder.pack(pady=10)

# Create a panel for displaying the image
panel = tk.Label(root)
panel.pack(pady=10)

# Create a frame with a scrollable text area for displaying the batch prediction results
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

scrollbar = tk.Scrollbar(result_frame, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

result_area = tk.Text(
    result_frame,
    yscrollcommand=scrollbar.set,
    wrap=tk.WORD,
    font=("Helvetica", 10),
    state=tk.DISABLED,
)
result_area.pack(expand=True, fill=tk.BOTH)

scrollbar.config(command=result_area.yview)

# Create a progress bar for displaying loading progress
progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=200, mode="determinate")
progress_bar.pack(pady=10)

# Run the application
root.mainloop()
