from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class LeafSchema(BaseModel):
    id: Optional[UUID] = None

    # 🌿 Plant information
    plant_type: str = Field(..., example="maize")
    disease_label: Optional[str] = Field(None, example="leaf_blight")
    health_status: Optional[str] = Field(None, example="healthy")  # healthy / stress / disease

    # 📸 Image info
    image_url: str
    image_name: Optional[str] = None

    # 📍 Location data
    gps_lat: Optional[float] = Field(None, example=9.5511)
    gps_lon: Optional[float] = Field(None, example=1.1860)
    location_name: Optional[str] = Field(None, example="Kara, Togo")

    # 👤 Collector info
    collector_id: Optional[str] = None
    device_info: Optional[str] = Field(None, example="Android - Samsung")

    # 🧠 AI / dataset usage
    is_annotated: Optional[bool] = False
    annotation_format: Optional[str] = Field(None, example="yolo")  # yolo / coco
    bounding_boxes: Optional[dict] = None  # pour YOLO annotations
    segmentation_mask_url: Optional[str] = None  # pour segmentation plus tard

    # ⏱ Metadata
    created_at: Optional[datetime] = None