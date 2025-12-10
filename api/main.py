"""FastAPI inference endpoint for SeasonsGAN."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
from io import BytesIO
from pathlib import Path

from src.models import Generator
from src.utils import load_image, tensor_to_image

app = FastAPI(title="SeasonsGAN API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model cache
_models = {}


def load_model(model_name: str) -> Generator:
    """Load a generator model from checkpoint."""
    if model_name in _models:
        return _models[model_name]
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = checkpoint_dir / f"{model_name}.pth"
    
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Model {model_name} not found at {checkpoint_path}"
        )
    
    gen = Generator(input_nc=3, output_nc=3, ngf=64, n_residual_blocks=9)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint)
    gen.to(device)
    gen.eval()
    
    _models[model_name] = gen
    return gen


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "device": str(device)}


@app.post("/transform")
async def transform_image(
    file: UploadFile = File(...), 
    direction: str = "summer2winter"
):
    """
    Transform an image using the specified generator.
    
    Args:
        file: Input image file
        direction: "summer2winter" or "winter2summer"
    
    Returns:
        Transformed image
    """
    if direction not in ["summer2winter", "winter2summer"]:
        raise HTTPException(status_code=400, detail="direction must be 'summer2winter' or 'winter2summer'")
    
    model_name = "gen_winter" if direction == "summer2winter" else "gen_summer"
    
    try:
        gen = load_model(model_name)
    except HTTPException:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_name} not available. Please ensure checkpoints are loaded."
        )
    
    # Read image
    try:
        image_bytes = await file.read()
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Convert to tensor
    image_tensor = load_image_from_pil(image_pil)
    
    # Inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output_tensor = gen(image_tensor)
    
    # Convert to PIL and return
    output_pil = tensor_to_image(output_tensor)
    
    # Save to BytesIO for response
    output_bytes = BytesIO()
    output_pil.save(output_bytes, format="PNG")
    output_bytes.seek(0)
    
    return FileResponse(
        iter([output_bytes.getvalue()]),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=output.png"}
    )


@app.get("/models")
def list_models():
    """List available models."""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        return {"models": [], "note": "No checkpoints directory found"}
    
    models = [f.stem for f in checkpoint_dir.glob("*.pth")]
    return {"models": models}


def load_image_from_pil(pil_image, size=256):
    """Load a PIL image and convert to tensor."""
    import numpy as np
    
    pil_image = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    img_array = np.array(pil_image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
