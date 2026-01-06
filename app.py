import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import pdfkit
from werkzeug.utils import secure_filename
import shutil

# TensorFlow settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Flask setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'malaria-detection-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# PDFKit configuration (Windows users)
WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"  # Update path if needed
pdf_config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

# Load model
print("Loading malaria model...")
try:
    model = load_model("model.keras")
    print("✓ Model loaded successfully")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print("✗ Model loading failed:", e)
    model = None

CLASS_LABELS = {
    0: "Parasitized",
    1: "Uninfected"
}

# Store last report info for PDF generation
last_report = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(240, 240))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_malaria(image_path):
    if model is None:
        return None, "Model not loaded"

    img = preprocess_image(image_path)
    preds = model.predict(img)[0]

    results = []
    for i, score in enumerate(preds):
        results.append({
            "class_id": i,
            "label": CLASS_LABELS[i],
            "confidence": float(score * 100)
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "prediction": results[0],
        "all_predictions": results
    }, None

# -------------------- Routes -------------------- #
@app.route("/")
def index():
    return render_template(
        "index.html",
        classes=CLASS_LABELS,
        model_loaded=model is not None
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    fullname = request.form.get("fullname", "John Doe")
    patient_id = request.form.get("patient_id", "PT-10001")

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save uploaded file to uploads folder
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    # Also save a copy in static folder for PDF inclusion
    static_path = os.path.join(app.config["STATIC_FOLDER"], "uploaded_scan.png")
    shutil.copy(upload_path, static_path)

    # Run prediction
    result, error = predict_malaria(upload_path)
    if error:
        return jsonify({"error": error}), 500

    # Save last report for PDF generation
    last_report.update({
        "fullname": fullname,
        "patient_id": patient_id,
        "prediction": result["prediction"],
        "image_path": static_path
    })

    result["filename"] = filename
    result["image_url"] = f"/{upload_path}"

    return jsonify(result)

@app.route("/download_report")
def download_report():
    if not last_report:
        return "No report available. Please scan a patient first.", 400

    # Generate HTML from template
    html = render_template(
        "report_template.html",
        fullname=last_report["fullname"],
        patient_id=last_report["patient_id"],
        prediction=last_report["prediction"]
    )

    pdf_path = "report.pdf"
    pdfkit.from_string(html, pdf_path, configuration=pdf_config)
    return send_file(pdf_path, as_attachment=True)

@app.route("/api/info")
def api_info():
    return jsonify({
        "model_loaded": model is not None,
        "input_shape": model.input_shape if model else None,
        "num_classes": len(CLASS_LABELS),
        "classes": CLASS_LABELS
    })

# -------------------- Main -------------------- #
if __name__ == "__main__":
    print("\n==============================")
    print("🦠 Malaria Cell Detection API")
    print("==============================")
    print("Classes:")
    for k, v in CLASS_LABELS.items():
        print(f" {k}: {v}")
    print("\nServer running at:")
    print("👉 http://127.0.0.1:5000")
    print("==============================\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
