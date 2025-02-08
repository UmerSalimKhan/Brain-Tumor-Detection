import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image  # Import PIL Image
import torchvision.models as models

# Loading the model
try:
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),  # Changed to ReLU
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.Tanh(), # Keep Tanh if it works better
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=torch.device('cpu'))) # Added map_location
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop() # Stop the app if the model can't be loaded

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode to NumPy array

        if image_np is None:
            st.error("Uploaded file could not be decoded as an image. Please upload a valid JPEG image.")
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # BGR to RGB

            image_pil = Image.fromarray(image_np)  # Convert NumPy array to PIL Image

            st.image(image_pil, caption="Uploaded Image", use_column_width=True)  # Display PIL Image

            if st.button("Detect"):
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),  # Now this will work!
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
                ])

                input_tensor = transform(image_pil).unsqueeze(0).to(device)  # Transform the PIL Image

                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = torch.sigmoid(output).item()

                if prediction > 0.5:
                    st.write("Detected")
                else:
                    st.write("Not Detected")

            if st.button("Reset"):
                uploaded_file = None

    except Exception as e:
        st.error(f"Error processing image: {e}")
