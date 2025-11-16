import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

model = load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Adjust if model uses a different size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    # Clean label
    label = CLASS_NAMES[top_idx]
    label = label.replace("plantvillage", "").replace("_", " ").strip()

    return {
        "label": label,
        "confidence": confidence
    }
