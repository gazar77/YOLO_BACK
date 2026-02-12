import os
import io
import json
import base64
import uuid
import datetime
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

# 1. تفعيل الـ CORS للربط مع Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. تحميل الموديل (تأكد من وجود الملف في نفس المسار)
try:
    model = YOLO("best (1).pt")
    print("✅ YOLO Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# 3. إعداد المجلدات والملفات
OS_RESULTS_DIR = "static/results"
os.makedirs(OS_RESULTS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

RESULTS_FILE = os.path.join(OS_RESULTS_DIR, "results.json")
HISTORY_FILE = os.path.join(OS_RESULTS_DIR, "history.json")

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# --- دالة الـ Preprocessing (كما ذُكر في البيبر) ---
def apply_clahe_preprocessing(img_np):
    # تحويل الصورة لـ Grayscale إذا كانت ملونة
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # تطبيق CLAHE لتحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # العودة لصيغة RGB لأن الموديل متدرب عليها
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # قراءة الصورة
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        if img_np is None:
            return JSONResponse({"error": "Invalid image format"}, status_code=400)

        # 4. تطبيق Preprocessing (CLAHE)
        processed_img = apply_clahe_preprocessing(img_np)

        # 5. عمل البريديكشن
        results = model(processed_img)

        # تحويل النتائج لـ JSON
        try:
            detections = json.loads(results[0].to_json())
        except:
            detections = []

        # 6. حفظ الصورة المرسومة (Annotated)
        unique_name = uuid.uuid4().hex
        annotated_name = f"{unique_name}_annotated.jpg"
        annotated_path = os.path.join(OS_RESULTS_DIR, annotated_name)

        # رسم البوكسات على الصورة الأصلية
        annotated_img = results[0].plot() 
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

        # 7. تحضير الـ Response
        output_data = {
            "id": unique_name,
            "date": str(datetime.datetime.now()),
            "annotated_image_url": f"/static/results/{annotated_name}",
            "detections": detections
        }

        # حفظ آخر نتيجة
        with open(RESULTS_FILE, "w") as f:
            json.dump(output_data, f, indent=4)

        # تحديث التاريخ (History)
        with open(HISTORY_FILE, "r") as f:
            history_data = json.load(f)
        
        history_data.append(output_data)
        
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_data, f, indent=4)

        return output_data

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    if not os.path.exists(HISTORY_FILE):
        return {"error": "History file not found"}

    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)

    # البحث عن الصورة لمسحها من الهارد
    item_to_delete = next((item for item in history_data if item["id"] == item_id), None)
    
    if item_to_delete:
        img_path = os.path.join(OS_RESULTS_DIR, os.path.basename(item_to_delete["annotated_image_url"]))
        if os.path.exists(img_path):
            os.remove(img_path)
        
        # تحديث قائمة التاريخ
        new_history = [item for item in history_data if item["id"] != item_id]
        with open(HISTORY_FILE, "w") as f:
            json.dump(new_history, f, indent=4)
        
        return {"message": "Deleted successfully"}
    
    return {"error": "Item not found"}

if __name__ == "__main__":
    import uvicorn
    # الكود هيشتغل على البورت اللي السيرفر محدده أو 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)