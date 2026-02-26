import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router  # importe ton router

# ============================================================
# Configuration
# ============================================================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Plant Disease Detection API",
    description="Hierarchical plant & disease detection system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://zaliapp.vercel.app"
    ],
    allow_credentials=False,  # Must be False when not using specific credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Inclure les routes
# ============================================================
app.include_router(router)

# ============================================================
# Lancer l'API avec uvicorn:
# uvicorn main:app --reload
# ============================================================