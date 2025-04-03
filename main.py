from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

# Load YOLO model (Ensure "best.pt" is in the working directory)
model = YOLO("best.pt")

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB PIL image
        img = np.array(img)  # Convert to NumPy
        
        # Run YOLO model on the image
        results = model(img)
        
        detections = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                label = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": class_id,
                    "label": label
                })

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Railway assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
