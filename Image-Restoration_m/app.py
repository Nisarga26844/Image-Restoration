import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
from natsort import natsorted
import cv2
import os
from runpy import run_path

# Define the image processing function
def process_image(uploaded_image, task):
    # Convert the uploaded image to an actual image file
    img = Image.open(uploaded_image).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Define model loading
    load_file = run_path(os.path.join(task, "MPRNet.py"))
    model = load_file['MPRNet']()
    model.cuda()

    weights = os.path.join(task, "pretrained_models", "model_" + task.lower() + ".pth")
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # Remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()

    # Pad the input if not multiple of 8
    img_multiple_of = 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)
    restored = restored[0]
    restored = torch.clamp(restored, 0, 1)

    # Unpad the output
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    return restored

# Create the Streamlit app
def main():
    st.title("Image Restoration Demo")
    st.sidebar.header("Select Task")

    # Task selection (Deblurring, Denoising, or Deraining)
    task = st.sidebar.selectbox('Select Task', ['Deblurring', 'Denoising', 'Deraining'])

    # File uploader for input image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Processing...")

        if st.button('Process'):
            # Process the image
            result_image = process_image(uploaded_image, task)

            # Display the result image
            st.image(result_image, caption="Processed Image", use_column_width=True)
            st.write("Processing complete!")

# Run the app
if __name__ == "__main__":
    main()
