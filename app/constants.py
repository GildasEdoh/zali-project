IMG_SIZE = (256, 256)
import os

# Base directory for the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models-cleaned")

PLANT_DISEASES = {
        'Apple': [
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___Apple_scab',
            'Apple___healthy'
        ],
        'Tomato': [
            'Tomato_Bacterial_spot',
            'Tomato_Early_blight',
            'Tomato_Late_blight',
            'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite',
            'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato__Tomato_mosaic_virus',
            'Tomato_healthy'
        ],
        'Potato': [
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy'
        ],
        'Corn_(maize)': [
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy'
        ],
        'Pepper': [
            'Pepper__bell___Bacterial_spot',
            'Pepper__bell___healthy'
        ]
    }

PLANT_MODEL_PATH = os.path.join(MODELS_DIR, "best_plant_classifier_clean.pth")
DISEASE_MODELS_PATH = MODELS_DIR

APPLE_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "apple_best_plant_classifier_clean.pth")
CORN_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "maize_best_plant_classifier_clean.pth")
PEPPER_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "pepper_best_plant_classifier_clean.pth")
TOMATO_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "tomato_best_plant_classifier_clean.pth")
POTATO_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "potato_best_plant_classifier_clean.pth")
