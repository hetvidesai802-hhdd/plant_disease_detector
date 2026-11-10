from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MODEL_PATH = Path("models/plant_disease_resnet18.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
class_names = []
preprocess = None

def load_model():
    global model, class_names, preprocess
    if not MODEL_PATH.exists():
        print("Model not found. Train it with `python train.py` before starting Flask.")
        return
    # This file is the locally generated checkpoint from train.py.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    preprocess = transforms.Compose([
        transforms.Resize((checkpoint.get("image_size", 224),
                           checkpoint.get("image_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        if model is None:
            error = "Model is not trained yet. Run prepare_data.py and train.py first."
        elif "leaf_image" not in request.files or not request.files["leaf_image"].filename:
            error = "Please choose a leaf image."
        else:
            uploaded = request.files["leaf_image"]
            if not allowed_file(uploaded.filename):
                error = "Please upload a JPG, JPEG, PNG, or WEBP image."
            else:
                try:
                    image = Image.open(uploaded.stream).convert("RGB")
                    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probabilities = torch.softmax(model(tensor), dim=1)[0]
                        confidence, index = probabilities.max(0)
                    label = class_names[index.item()]
                    crop, disease = label.split("___", 1) if "___" in label else ("Plant", label)
                    prediction = {
                        "crop": crop.replace("_", " "),
                        "disease": disease.replace("_", " "),
                        "confidence": round(confidence.item() * 100, 2),
                        "healthy": "healthy" in disease.lower(),
                    }
                except (UnidentifiedImageError, OSError):
                    error = "That file could not be read as an image."
    return render_template("index.html", prediction=prediction, error=error)

@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="Image is too large. Maximum size is 8 MB."), 413

load_model()

if __name__ == "__main__":
    app.run(debug=True)
