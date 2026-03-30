import os
import json
from google import genai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load .env here so the key is always available regardless of import order
load_dotenv()

class GeminiService:
    def __init__(self):
        self._model = None

    def _get_model(self):
        """Lazy initialization — reads API key only when first needed."""
        if self._model is not None:
            return self._model
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print(f"DEBUG: Gemini API Key found (starts with {api_key[:5]})")
            client = genai.Client(api_key=api_key)
            self._model = client
        else:
            print("DEBUG: Gemini API Key NOT found in environment")
        return self._model

    def generate_plant_recommendations(self, plant_name: str, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate personalised recommendations via Gemini.
        """
        model = self._get_model()
        if not model:
            print("DEBUG: No model available, skipping Gemini.")
            return {}

        if disease_name and "healthy" not in disease_name.lower():
            prompt = f"""
            As an expert agronomist, provide a technical and concise analysis for:
            Plant: {plant_name}
            Detected Disease: {disease_name}
            
            Return ONLY a JSON object with:
            - recommendation: Brief technical diagnostic and immediate action (max 2 sentences).
            - water_needed_mm: Numerical irrigation volume (e.g., 20).
            - frequency: Technical irrigation frequency (e.g., 'Every 48h').
            - best_watering_time: Best time of day to water this plant given its disease (e.g., 'Early morning, 6-8 AM').
            - notes: List of 3 concise, science-based protocols for treatment.
            
            Language: English. No markdown, no preamble.
            """
        else:
            prompt = f"""
            As an expert agronomist, provide optimized maintenance for:
            Plant: {plant_name}
            Status: Healthy.
            
            Return ONLY a JSON object with:
            - recommendation: Brief technical growth optimization tip (max 1 sentence).
            - water_needed_mm: Numerical baseline irrigation volume (e.g., 25).
            - frequency: Maintenance irrigation frequency (e.g., 'Every 72h').
            - best_watering_time: Best time of day to water this plant (e.g., 'Early morning, 6-8 AM').
            - notes: List of 3 concise best practices for health maintenance.
            
            Language: English. No markdown, no preamble.
            """

        try:
            print(f"DEBUG: Sending request to Gemini for {plant_name} / {disease_name}...")
            response = model.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            print("DEBUG: Gemini response received.")
            
            text = response.text.strip()
            # Extract JSON robustly
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            print("DEBUG: JSON successfully parsed.")
            return result
        except Exception as e:
            print(f"DEBUG Error calling Gemini or parsing JSON: {e}")
            if 'text' in locals():
                print(f"DEBUG Raw text was: {text[:200]}")
            return {}

_service: Optional[GeminiService] = None

def get_gemini_service() -> GeminiService:
    global _service
    if _service is None:
        _service = GeminiService()
    return _service
