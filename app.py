import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

app = Flask(__name__)

# -------------------------------
# CONFIG
# -------------------------------

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224
NUM_CLASSES = 7

# -------------------------------
# CLASS LABELS (MUST MATCH TRAINING)
# -------------------------------

class_labels = [
    'Aeromoniasis',
    'Bacterial Red disease',
    'Bacterial gill disease',
    'Fungal Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral White tail disease'
]

# -------------------------------
# DISEASE INFORMATION
# -------------------------------

disease_info = {
    "Healthy Fish": {
        "water_temp": "Optimal",
        "medicine": "None",
        "suggestion": "Maintain normal water quality and feeding schedule."
    },
    "Aeromoniasis": {
        "water_temp": "24-26°C",
        "medicine": "Oxytetracycline",
        "suggestion": "Isolate infected fish and improve water hygiene."
    },
    "Bacterial Red disease": {
        "water_temp": "24-26°C",
        "medicine": "Oxytetracycline",
        "suggestion": "Improve water quality and reduce stress."
    },
    "Bacterial gill disease": {
        "water_temp": "24-26°C",
        "medicine": "Oxytetracycline",
        "suggestion": "Increase aeration and reduce stocking density."
    },
    "Fungal Saprolegniasis": {
        "water_temp": "25°C",
        "medicine": "Methylene Blue",
        "suggestion": "Remove dead tissue and disinfect tank."
    },
    "Parasitic diseases": {
        "water_temp": "28°C",
        "medicine": "Formalin treatment",
        "suggestion": "Quarantine infected fish immediately."
    },
    "Viral White tail disease": {
        "water_temp": "26-28°C",
        "medicine": "No effective medicine",
        "suggestion": "Preventive care and strict isolation required."
    }
}

# -------------------------------
# REBUILD MODEL EXACTLY LIKE COLAB
# -------------------------------

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=None   # IMPORTANT
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Build model before loading weights
model.build((None, IMG_SIZE, IMG_SIZE, 3))

# Load CLEAN weights exported from Colab
model.load_weights("fish_weights_clean.weights.h5")

print("✅ Model loaded successfully")

# -------------------------------
# ROUTES
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/technology")
def technology():
    return render_template("technology.html")

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("detection"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # SAME preprocessing used in training
    img = Image.open(filepath).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = round(float(np.max(predictions)) * 100, 2)

    predicted_class = class_labels[predicted_index]
    info = disease_info.get(predicted_class, {})

    return render_template(
        "detection.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=filepath,
        medicine=info.get("medicine", "N/A"),
        temperature=info.get("water_temp", "N/A"),
        advice=info.get("suggestion", "N/A")
    )

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
