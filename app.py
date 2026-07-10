import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import onnxruntime as ort

app = Flask(__name__)

# Load model ONNX
model_path = "vgg_cataract.onnx"
session = ort.InferenceSession(model_path)

# Parameter preprocessing (sama dengan notebook)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Nama kelas (urut sesuai folder: Cataract, Normal)
CLASS_NAMES = ["Cataract", "Normal"]

def preprocess_image(image):
    """
    Menerima PIL Image, mengembalikan numpy array siap untuk inferensi.
    """
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    # Convert ke array dan normalisasi ke [0,1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Jika gambar grayscale, konversi ke RGB
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    # Transpose ke (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    # Normalisasi dengan mean dan std ImageNet
    for i in range(3):
        img_array[i] = (img_array[i] - MEAN[i]) / STD[i]
    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File kosong'}), 400

    try:
        # Baca gambar
        image = Image.open(file.stream).convert('RGB')
        # Preprocess
        input_tensor = preprocess_image(image)

        # Inferensi
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_tensor})
        probs = result[0][0]  # shape: (2,)

        # Konversi ke probabilitas (softmax)
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / np.sum(exp_probs)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        label = CLASS_NAMES[pred_class]

        return jsonify({
            'label': label,
            'confidence': confidence,
            'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)