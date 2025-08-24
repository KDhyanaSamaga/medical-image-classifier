from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("medical_mnist_model.h5")

# Class names (must match your dataset folders)
classes = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]

# Home route -> serve frontend
@app.route("/")
def index():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # Preprocess image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, 64, 64, 1) / 255.0

        # Predict
        pred = model.predict(img)
        class_index = np.argmax(pred)
        result = classes[class_index]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Remove file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"prediction": result}), 200


if __name__ == "__main__":
    app.run(debug=True)
