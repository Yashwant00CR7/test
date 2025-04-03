from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")

def process_image(image):
    """Run YOLO model on the image and return predictions."""
    results = model(image)
    detections = []
    
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class": int(cls),
                "label": model.names[int(cls)]
            })
    
    return detections

@app.route('/predict', methods=['POST'])
def predict():
    """Receive an image, process it, and return YOLO detections."""
    try:
        data = request.json['image']
        image_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)

        results = process_image(image)

        return jsonify({"detections": results})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
