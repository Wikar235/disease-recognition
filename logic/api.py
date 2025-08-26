from fastapi import FastAPI, UploadFile, File
import os
import shutil
from predict import load_model, predict

MODEL_PATH = os.getenv("MODEL_PATH", "runs/train/weights/best.pt")
model = load_model(MODEL_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Disease recognition API is running!"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = predict(model, temp_file)

    os.remove(temp_file)
    return {"detections": detections}
