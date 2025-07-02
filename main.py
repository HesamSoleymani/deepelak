from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import keras_cv

app = FastAPI()

print("Loading models...")
plate_detector = keras.models.load_model("./models/car_model.keras")
plate_recognizer = keras.models.load_model("./models/plate_model.keras")
print("Models loaded successfully!")

class_mapping = {0: '9', 1: '1', 2: 'و', 3: '2', 4: '6', 5: '7', 6: '8', 7: 'ن', 8: '3', 9: '5', 10: 'م', 11: 'ی', 12: 'ت', 13: 'ل', 14: '4', 15: 'ه\u200d', 16: 'ط', 17: '0', 18: 'د', 19: 'ق', 20: 'ص', 21: 'ب', 22: 'ج', 23: 'س', 24: 'ع', 25: 'ژ (معلولین و جانبازان)', 26: 'الف', 27: 'ز', 28: 'ش', 29: 'پ', 30: 'ث'}

def extract_license_text(prediction):
    boxes = prediction['boxes'][0]
    classes = prediction['classes'][0]
    confidences = prediction['confidence'][0]
    num_detections = prediction['num_detections'][0]
    
    char_boxes = []
    for i in range(num_detections):
        if classes[i] < 0 or confidences[i] < 0:
            continue
        char_box = boxes[i]
        char_boxes.append((char_box, classes[i]))
    
    char_boxes.sort(key=lambda x: x[0][0])
    chars = [class_mapping[int(c)] for _, c in char_boxes]
    return ''.join(chars)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict_license_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        image_tensor = tf.convert_to_tensor(np.array(image))
        
        h, w = image_tensor.shape[0], image_tensor.shape[1]
        if h > w:
            w1 = w * 640 / h
            h1 = 640
        else:
            h1 = h * 640 / w
            w1 = 640
        
        image_resized = tf.image.resize_with_pad(image_tensor, target_height=640, target_width=640)
        
        plate_pred = plate_detector.predict(np.array([image_resized]))
        
        results = []
        
        for i in range(plate_pred["num_detections"][0]):
            box = plate_pred["boxes"][0][i]
            confidence = plate_pred["confidence"][0][i]
            
            if confidence < 0.5:
                continue
                
            x1 = int((box[0] - (640 - w1) / 2) * max(w, h) / 640)
            x2 = int((box[2] - (640 - w1) / 2) * max(w, h) / 640)
            y1 = int((box[1] - (640 - h1) / 2) * max(w, h) / 640)
            y2 = int((box[3] - (640 - h1) / 2) * max(w, h) / 640)
            
            plate_region = image_tensor[y1:y2, x1:x2]
            plate_resized = tf.image.resize_with_pad(plate_region, target_height=640, target_width=640)

            char_pred = plate_recognizer.predict(np.array([plate_resized]))
            license_text = extract_license_text(char_pred)

            keras_cv.visualization.plot_bounding_box_gallery(
                images=np.array([plate_resized]),
                value_range=(0, 255),
                bounding_box_format="xyxy",
                y_pred=char_pred,
                scale=7,
                rows=1,
                cols=1,
                path="plate.png",
                font_scale=0.7,
            )
            
        #     results.append({
        #         "license_text": license_text,
        #         "confidence": float(confidence),
        #         "bbox": [int(x1), int(y1), int(x2), int(y2)]
        #     })
        
        # return {"results": results}
        return FileResponse(
            "plate.png",
            media_type='image/png',
            filename='plot.png'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")