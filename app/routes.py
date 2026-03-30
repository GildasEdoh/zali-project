import os
import io
import base64
from io import BytesIO
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import uuid
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from PIL import Image
from datetime import datetime

from hierarchical_desease_classifier import HierarchicalPlantDiseaseDetector
from classifiers.plant_classifier import get_classifier, PlantClassifier
from hierarchical_desease_classifier import get_desease_classifier
from constants import PLANT_MODEL_PATH, DISEASE_MODELS_PATH, IMG_SIZE
from schemas import LeafSchema
from dotenv import load_dotenv

from supabase import create_client, Client
import os

# ============================================================
# Router
# ============================================================
load_dotenv()
router = APIRouter()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("SUPABASE_URL", SUPABASE_URL)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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


@router.post("/upload_leaf")
async def upload_leaf(
    plant_type: str = Form(...),
    disease_label: str = Form(None),
    health_status: str = Form(None),

    gps_lat: float = Form(None),
    gps_lon: float = Form(None),
    location_name: str = Form(None),

    collector_id: str = Form(None),
    device_info: str = Form(None),

    image: UploadFile = File(...)
):
    # 1. nom unique
    filename = f"{uuid.uuid4()}.jpg"

    file_bytes = await image.read()

    # --- TRAITEMENT DE L'IMAGE ---
    # 1. Ouvrir l'image depuis les bytes
    img = Image.open(io.BytesIO(file_bytes))

    if img.mode != "RGB":
        img = img.convert("RGB")


    # 2. Redimensionner (256x256)
    # L'utilisation de LANCZOS garantit une meilleure qualité après réduction
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    # 3. Convertir l'image traitée en bytes pour Supabase
    buffer = io.BytesIO()
    # On force le format JPEG (ou utilisez img.format)
    img.save(buffer, format="JPEG", quality=85)
    resized_bytes = buffer.getvalue()

    # 3. upload Supabase (Utilisez resized_bytes au lieu de file_bytes)
    path = f"{plant_type}/{filename}"

    supabase.storage.from_("leaf-images").upload(
        path,
        resized_bytes,
        {
            "content-type": "image/jpeg",
            "upsert": "true"
        }
    )

    # 4. URL publique
    image_url = supabase.storage.from_("leaf-images").get_public_url(path)

    # 5. save DB
    supabase.table("leaf_images").insert({
        "plant_type": plant_type,
        "disease_label": disease_label,
        "health_status": health_status,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "location_name": location_name,
        "collector_id": collector_id,
        "device_info": device_info,
        "image_url": image_url,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    return {
        "message": "Image uploaded and resized successfully",
        "image_url": image_url
    }

@router.get("/leaf_images")
async def get_all_leaf_images():
    
    response = supabase.table("leaf_images").select("*").execute()

    return {
        "count": len(response.data),
        "data": response.data
    }

@router.get("/leaf_image")
async def get_leaf_image(image_url: str):
    return RedirectResponse(image_url)