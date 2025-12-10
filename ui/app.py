"""Streamlit UI for SeasonsGAN Image Translation."""
import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from src.models import Generator
from src.utils import load_image, tensor_to_image

st.set_page_config(page_title="SeasonsGAN", layout="wide", initial_sidebar_state="expanded")

st.title("🌞❄️ SeasonsGAN: Season Translation")
st.write(
    "Upload an image to transform between Summer and Winter seasons using a trained CycleGAN model."
)

# Sidebar: Configuration
st.sidebar.header("Configuration")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"**Device:** {device}")

direction = st.sidebar.radio(
    "Select transformation direction:",
    options=["Summer → Winter", "Winter → Summer"],
    index=0
)

model_mapping = {
    "Summer → Winter": "gen_winter",
    "Winter → Summer": "gen_summer",
}
model_name = model_mapping[direction]

# Load model
@st.cache_resource
def load_generator_model(name):
    checkpoint_path = Path("SeasonsGAN/checkpoints") / f"{name}.pth"
    
    if not checkpoint_path.exists():
        st.warning(f"⚠️ Model checkpoint not found at {checkpoint_path}")
        st.info(
            "Please ensure the model weights are saved to `SeasonsGAN/checkpoints/` "
            "after training completes on Colab."
        )
        return None
    
    gen = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=9)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint)
    gen.to(device)
    gen.eval()
    return gen


# File upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image (JPG, PNG):",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load input image
    input_image = Image.open(uploaded_file).convert("RGB")
    
    # Display input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_column_width=True)
    
    # Load model and perform inference
    with col2:
        st.subheader("Output Image")
        
        gen = load_generator_model(model_name)
        
        if gen is not None:
            # Show loading spinner
            with st.spinner("Transforming image..."):
                # Prepare input
                input_tensor = load_image(uploaded_file.name, size=256)
                
                # Inference
                with torch.no_grad():
                    input_tensor = input_tensor.to(device)
                    output_tensor = gen(input_tensor)
                
                # Convert to PIL
                output_image = tensor_to_image(output_tensor)
                st.image(output_image, use_column_width=True)
                
                # Download button
                output_bytes = np.array(output_image)
                st.download_button(
                    label="📥 Download Result",
                    data=Image.fromarray(output_bytes).tobytes(),
                    file_name="transformed.png",
                    mime="image/png"
                )
else:
    st.info("👈 Upload an image to get started!")

# Footer
st.divider()
st.markdown(
    """
    **About SeasonsGAN**
    
    This app uses a CycleGAN model trained on Yosemite Summer ↔ Winter dataset.
    
    - **Model:** CycleGAN with 9 residual blocks
    - **Training Data:** Unpaired summer and winter landscape images
    - **Framework:** PyTorch
    """
)
