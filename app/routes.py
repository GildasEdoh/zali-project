import os
import base64
from io import BytesIO
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from hierarchical_desease_classifier import HierarchicalPlantDiseaseDetector
from classifiers.plant_classifier import get_classifier, PlantClassifier
from hierarchical_desease_classifier import get_desease_classifier
from constants import PLANT_MODEL_PATH, DISEASE_MODELS_PATH

# ============================================================
# Router
# ============================================================
router = APIRouter()

# ============================================================
# Configuration / Model paths
# ============================================================


detector: HierarchicalPlantDiseaseDetector = None

# ============================================================
# Startup event to load model
# ============================================================
@router.on_event("startup")
def load_model():
    global detector
    print("Loading models...")
    print("Models loaded successfully!")

# ============================================================
# Base64 schema
# ============================================================
class Base64Image(BaseModel):
    image: str

# ============================================================
# Routes
# ============================================================
@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/info")
def system_info():
    return {
        "plant_model": PLANT_MODEL_PATH,
        "disease_models_path": DISEASE_MODELS_PATH
    }


@router.post("/predict_plant_class")
async def predict_plant_class(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    sample_classifier : PlantClassifier = get_classifier(PLANT_MODEL_PATH)
    result = sample_classifier.predict_img(image)
    return JSONResponse(content=result)

@router.post("/predict_plant_desease")
async def predict_plant_desease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    detector : HierarchicalPlantDiseaseDetector = get_desease_classifier()
    result = detector.predict_desease(image)

    return JSONResponse(content=result)


# @router.post("/predict-batch")
# async def predict_batch(files: List[UploadFile] = File(...)):
#     results = []
#     for file in files:
#         contents = await file.read()
#         image = Image.open(BytesIO(contents)).convert("RGB")
#         result = detector.predict(image)
#         results.append(result)
#     return JSONResponse(content=results)


# @router.post("/predict-base64")
# def predict_base64(data: Base64Image):
#     try:
#         image_data = base64.b64decode(data.image)
#         image = Image.open(BytesIO(image_data)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid base64 image")

#     result = detector.predict(image)
#     return JSONResponse(content=result)