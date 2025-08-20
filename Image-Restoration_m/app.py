import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
import os
from runpy import run_path

# ============================
# Image Processing Function
# ============================
def process_image(uploaded_image, task):
    # Convert uploaded file to PIL image
    img = Image.open(uploaded_image).convert('RGB')

    # 🔧 Resize if image too large (avoid CUDA OOM)
    max_size = 720   # try 512 if still OOM
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))   # keeps aspect ratio

    # Convert to tensor and send to GPU
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Free any cached memory
    torch.cuda.empty_cache()

    # Load the model dynamically
    load_file = run_path(os.path.join(task, "MPRNet.py"))
    model = load_file['MPRNet']().cuda()

    weights = os.path.join(task, "pretrained_models", "model_" + task.lower() + ".pth")
    checkpoint = torch.load(weights)

    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        # Handle if checkpoint keys have 'module.' prefix
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()

    # Pad input if not multiple of 8
    img_multiple_of = 8
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh, padw = H - h, W - w
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # 🔧 Use mixed precision to save memory
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            restored = model(input_)

    restored = restored[0]
    restored = torch.clamp(restored, 0, 1)

    # Remove padding
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    return restored

# ============================
# Streamlit App
# ============================
def main():
    st.title("Image Restoration Demo")
    st.sidebar.header("Select Task")

    # Select task
    task = st.sidebar.selectbox('Select Task', ['Deblurring', 'Denoising', 'Deraining'])

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Processing...")

        if st.button('Restore Image'):
            result_image = process_image(uploaded_image, task)

            # Show result
            st.image(result_image, caption="Processed Image", use_column_width=True)
            st.write("Restored Image")

# ============================
# Run Streamlit App
# ============================
if __name__ == "__main__":
    main()
