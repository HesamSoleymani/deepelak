from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import keras_cv
import base64
import uuid

app = FastAPI()

print("Loading models...")
plate_detector = keras.models.load_model("./models/car_model.keras")
plate_recognizer = keras.models.load_model("./models/plate_model.keras")
print("Models loaded successfully!")

class_mapping = {0: '9', 1: '1', 2: 'و', 3: '2', 4: '6', 5: '7', 6: '8', 7: 'ن', 8: '3', 9: '5', 10: 'م', 11: 'ی', 12: 'ت', 13: 'ل', 14: '4', 15: 'ه\u200d', 16: 'ط', 17: '0', 18: 'د', 19: 'ق', 20: 'ص', 21: 'ب', 22: 'ج', 23: 'س', 24: 'ع', 25: 'ژ (معلولین و جانبازان)', 26: 'الف', 27: 'ز', 28: 'ش', 29: 'پ', 30: 'ث'}

temp_storage = {}

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

@app.post("/detect")
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

        conf_threshold = 0.5

        conf = plate_pred["confidence"][0]
        boxes = plate_pred["boxes"][0]
        classes = plate_pred["classes"][0]
        valid_count = 0

        for i in range(len(conf)):
            if conf[i] < conf_threshold:
                conf[i] = -1
                classes[i] = -1
                boxes[i] = [-1, -1, -1, -1]
            else:
                valid_count += 1

        plate_pred["num_detections"][0] = valid_count        

        rid = str(uuid.uuid4())
        filename = "car_" + rid + ".png"

        keras_cv.visualization.plot_bounding_box_gallery(
            images=np.array([image_resized]),
            value_range=(0, 255),
            bounding_box_format="xyxy",
            y_pred=plate_pred,
            scale=7,
            rows=1,
            cols=1,
            path=filename,
        )

        with open(filename, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        temp_storage[rid] = {
            "index": 0,
            "pred": plate_pred,
            "sizes": [w,h,w1,h1],
            "image_tensor": image_tensor,
        }

        return JSONResponse(content={
            "image": image_base64,
            "rid": rid
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error : {str(e)}")

@app.get("/recognize")
async def predict_license_plate(rid:str):
    if not rid in temp_storage:
        raise HTTPException(status_code=400, detail="invalid rid")
    
    try:
        i = temp_storage[rid]["index"]
        plate_pred = temp_storage[rid]["pred"]
        box = plate_pred["boxes"][0][i]
        image_tensor = temp_storage[rid]["image_tensor"]
        w,h,w1,h1 = temp_storage[rid]["sizes"]
        filename = "plate_" + rid + ".png"
            
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
            path=filename,
        )

        with open(filename, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        temp_storage[rid]["index"] = i + 1

        return JSONResponse(content={
            "image": image_base64,
            "more": bool(plate_pred["num_detections"][0] != temp_storage[rid]["index"])
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error : {str(e)}")