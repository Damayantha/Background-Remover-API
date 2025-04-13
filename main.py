import os
from io import BytesIO
from typing import List

import gdown
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from torchvision import transforms
import zipfile

# Import DIS model
from data_loader_cache import normalize, im_reader, im_preprocess
from models import ISNetDIS

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download and load model
if not os.path.exists("./saved_models/isnet.pth"):
    os.makedirs("./saved_models", exist_ok=True)
    MODEL_PATH_URL = "https://drive.google.com/uc?id=1XHIzgTzY5BQHw140EDIgwIb53K659ENH"
    gdown.download(MODEL_PATH_URL, "./saved_models/isnet.pth", use_cookies=False)

hypar = {
    "model_path": "./saved_models",
    "restore_model": "isnet.pth",
    "model_digit": "full",
    "cache_size": [1024, 1024],
    "input_size": [1024, 1024],
    "crop_size": [1024, 1024],
    "model": ISNetDIS()
}


# Build model function
def build_model(hypar, device):
    net = hypar["model"]
    net.to(device)
    net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
    net.eval()
    return net


net = build_model(hypar, device)


def preprocess_image(image: BytesIO):
    im = im_reader(image)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    transform = transforms.Compose([lambda x: normalize(x, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])
    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def predict_mask(image_tensor, orig_size):
    inputs_val = image_tensor.to(device).float()
    ds_val = net(inputs_val)[0]  # Get model output
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.interpolate(pred_val.unsqueeze(0), (orig_size[0][0], orig_size[0][1]), mode='bilinear'))
    pred_val = (pred_val - torch.min(pred_val)) / (torch.max(pred_val) - torch.min(pred_val))
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)


def process_single_image(image_bytes):
    """Process a single image and remove background"""
    image = Image.open(image_bytes).convert("RGBA")
    image_bytes.seek(0)  # Reset file pointer
    image_tensor, orig_size = preprocess_image(image_bytes)

    # Predict the mask and remove the background
    mask = predict_mask(image_tensor, orig_size)
    mask = Image.fromarray(mask).resize(image.size, Image.LANCZOS)

    # Convert mask to an alpha channel
    alpha = mask.convert("L")

    # Apply transparency by adding alpha channel to the original image
    transparent_img = Image.new("RGBA", image.size)
    transparent_img.paste(image, mask=alpha)

    return transparent_img


@app.post("/remove_bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = BytesIO(await file.read())
        transparent_img = process_single_image(image_bytes)

        # Save the transparent image to a byte stream and return
        img_io = BytesIO()
        transparent_img.save(img_io, format="PNG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_remove_bg")
async def batch_remove_background(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Create a zip file in memory to store processed images
        zip_io = BytesIO()
        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, file in enumerate(files):
                # Read the image file
                image_bytes = BytesIO(await file.read())
                transparent_img = process_single_image(image_bytes)

                # Save to zip
                img_io = BytesIO()
                transparent_img.save(img_io, format="PNG")
                img_io.seek(0)

                # Get original filename without extension and add .png
                original_filename = os.path.splitext(file.filename)[0]
                zip_file.writestr(f"{original_filename}_transparent.png", img_io.getvalue())

        zip_io.seek(0)
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=transparent_images.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional endpoint with more control over batch processing
@app.post("/batch_remove_bg_with_options")
async def batch_remove_bg_with_options(
        files: List[UploadFile] = File(...),
        batch_size: int = Form(8)  # Process images in batches of this size
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Create a zip file in memory to store processed images
        zip_io = BytesIO()
        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Process images in batches to avoid memory issues
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]

                for file in batch_files:
                    # Read the image file
                    image_bytes = BytesIO(await file.read())
                    transparent_img = process_single_image(image_bytes)

                    # Save to zip
                    img_io = BytesIO()
                    transparent_img.save(img_io, format="PNG")
                    img_io.seek(0)

                    # Get original filename without extension and add .png
                    original_filename = os.path.splitext(file.filename)[0]
                    zip_file.writestr(f"{original_filename}_transparent.png", img_io.getvalue())

        zip_io.seek(0)
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=transparent_images.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)