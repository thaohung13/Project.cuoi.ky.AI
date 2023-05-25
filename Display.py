import cv2
import tkinter as tk 
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model_path = 'MODEL1.h5'
model = load_model(model_path)

# Define class labels
class_labels = ['Hoa Hong','Hoa Su','Hoa Sen','Hoa Mai','Hoa Phuong Lu Do','Hoa Hong Mon','Hoa Thien Dieu','Hoa Cuc Van Tho','Hoa Huong Duong','Hoa Da Uyen Thao']  # Replace with actual class labels

# Create a window
window = tk.Tk()
window.title("Flower Sign Classification")

# Create a label to display the video feed
label = tk.Label(window)
label.pack()

# Open the webcam
cap = cv2.VideoCapture(0)

def update_video():
    # Read frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Perform flower recognition on the frame
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255.0  # Normalize the image
        reshaped_frame = np.expand_dims(normalized_frame, axis=0)
        prediction = model.predict(reshaped_frame)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw the predicted class label on the frame
        cv2.putText(frame_rgb, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Resize the frame to fit the label
        frame_resized = cv2.resize(frame_rgb, (640, 480))

        # Create an ImageTk object from the resized frame
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the label with the new frame
        label.imgtk = img_tk
        label.configure(image=img_tk)

    # Schedule the next update
    label.after(10, update_video)

# Start updating the video feed
update_video()

# Run the GUI main loop
window.mainloop()

# Release the webcam
cap.release()