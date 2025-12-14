"""
backend API for MedAI application using fastAPI framework
"""

import io
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from models.ct_mri_cycle_gans import Generator, Discriminator, weights_init,  val_transforms
import os
import torch
import base64
from io import BytesIO
import PIL
import numpy as np
import uvicorn


class CTImagePayload(BaseModel):
    ct_image: str  # base64 encoded string
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("Starting up the MedAI backend API. Loading models ...")
    generator = Generator()
    discriminator = Discriminator()
    discriminator.apply(weights_init)
    weights_dir = "models/pretrained_weights/"
    generator_weights_path = os.path.join(weights_dir, "generator.pth")
    discriminator_weights_path = os.path.join(weights_dir, "discriminator.pth")
    if os.path.exists(generator_weights_path):
        generator.load_state_dict(
            torch.load(generator_weights_path, map_location=torch.device('cpu'))
        )
        print("Loaded pre-trained Generator weights.")
    if os.path.exists(discriminator_weights_path):
        discriminator.load_state_dict(
            torch.load(discriminator_weights_path, map_location=torch.device('cpu'))
        )
        print("Loaded pre-trained Discriminator weights.")
    app.state.generator = generator
    app.state.discriminator = discriminator

    yield
    # Shutdown actions
    print("Shutting down the MedAI backend API...")

app = FastAPI(lifespan=lifespan,
              title="MedAI Backend API",
              description="API for Medical Image Translation using GANs",
              version="1.0.0")
@app.get("/")
def read_root():
    return {"message": "Welcome to the MedAI Backend API for Medical Image Translation using GANs."}

@app.get("/models") # Endpoint to get model summaries
def get_models():
    return {"generator": str(app.state.generator),
            "discriminator": str(app.state.discriminator)}

@app.get("/info")
def get_info():
    return ({"app": "MedAI Backend API",
             "version": "1.0.0",
             "description": "API for Medical Image Translation using GANs",
             "author": "Ha Pham"})

@app.post("/generate_mri")  # Endpoint to generate MRI from CT image
def generate_mri(payload: CTImagePayload):
    """
    Translate CT image to MRI using the pre-trained Generator model.
    param payload: CTImagePayload
    return {"mri_image": "base64_encoded_mri_image_string"}
    """
    # Decode the base64 encoded CT image
    img_bytes = base64.b64decode(payload.ct_image)
    # load image bytes into PIL image
    pil_ct_image = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # save the received image for debugging
    pil_ct_image.save("received_ct_image.png")

    # Preprocess the image (resize, normalize, etc.)
    ct_image_tensor = val_transforms(pil_ct_image).unsqueeze(0)  # Add batch dimension

    # Inference: Generate MRI image using the Generator model
    generator = app.state.generator
    generator.eval()
    with torch.no_grad():
        mri_image_tensor = generator(ct_image_tensor)
    
    # Convert the output tensor to a PIL image
    mri_array = mri_image_tensor.squeeze().cpu().numpy()
    # denormalizes the generator outputs from [-1, 1] to [0, 255]
    mri_array = ((mri_array + 1) / 2.0 * 255).astype('uint8')
    mri_pil = PIL.Image.fromarray(np.transpose(mri_array, (1, 2, 0)))  # Convert from CHW to HWC

    # Save the generated image for debugging
    mri_pil.save("generated_mri_image.png")

    # Encode the PIL image to base64 string
    buf = io.BytesIO()
    mri_pil.save(buf, format ='PNG')
    # reset buffer position to start
    buf.seek(0)

    img_bytes = buf.getvalue()
    mri_image_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return {"mri_image": mri_image_b64} 

# if __name__ == "__main__":
#     sample_image_path = sys.argv[1]
# if not os.path.exists(sample_image_path):
#     raise SystemExit(f"Input image not found: {sample_image_path}")

# # 2) create and load models (your existing code)
# generator = Generator()
# discriminator = Discriminator()
# discriminator.apply(weights_init)

# weights_dir = "models/pretrained_weights/"
# generator_weights_path = os.path.join(weights_dir, "generator.pth")
# discriminator_weights_path = os.path.join(weights_dir, "discriminator.pth")

# if os.path.exists(generator_weights_path):
#     generator.load_state_dict(torch.load(generator_weights_path, map_location="cpu"))
#     print("Loaded pre-trained Generator weights.")
# else:
#     print("Warning: generator weights not found:", generator_weights_path)

# if os.path.exists(discriminator_weights_path):
#     discriminator.load_state_dict(torch.load(discriminator_weights_path, map_location="cpu"))
#     print("Loaded pre-trained Discriminator weights.")
# else:
#     print("Warning: discriminator weights not found:", discriminator_weights_path)

# # 3) device handling
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator.to(device)
# generator.eval()

# # 4) load and preprocess image
# pil_ct_image = PIL.Image.open(sample_image_path).convert("RGB")
# ct_image_tensor = val_transforms(pil_ct_image).unsqueeze(0)  # shape: (1,C,H,W)
# ct_image_tensor = ct_image_tensor.to(device)

# # 5) inference
# with torch.no_grad():
#     out_tensor = generator(ct_image_tensor)  # expected shape (1,C,H,W)

# # 6) move to CPU, squeeze, permute to HWC, denormalize, clip and convert
# out_cpu = out_tensor.squeeze(0).cpu()              # shape (C,H,W)
# out_np = out_cpu.permute(1, 2, 0).numpy()          # shape (H,W,C)

# # assume generator uses Tanh -> outputs in [-1,1]
# out_np = (out_np + 1.0) / 2.0 * 255.0
# out_np = np.clip(np.round(out_np), 0, 255).astype(np.uint8)

# # 7) save image
# basename = os.path.basename(sample_image_path)
# save_path = f"generated_{basename}"
# PIL.Image.fromarray(out_np).save(save_path)
# print("Saved generated image:", save_path)