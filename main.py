
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Define the class names
CLASS_NAMES = [
    'dog', 'cat', 'horse', 'spider', 'butterfly',
    'chicken', 'sheep', 'cow', 'squirrel', 'elephant'
]

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

# Load the model
MODEL_PATH = "model_scripted_efficientnet_lr0.001_aughigh.pt"
try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and transform image
        img = Image.open(io.BytesIO(await file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Run prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_prob, top_class = torch.max(probs, 0)

        # Return prediction
        return JSONResponse({
            "prediction": CLASS_NAMES[top_class.item()],
            "confidence": round(top_prob.item() * 100, 2),
            "class_probabilities": {
                CLASS_NAMES[i]: round(p.item() * 100, 2)
                for i, p in enumerate(probs)
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
