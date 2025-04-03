from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")  # Ensure "best.pt" is in the same folder

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))  # Convert to PIL
        img = np.array(img)  # Convert to numpy
        
        results = model(img)  # Run YOLO
        
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

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}
