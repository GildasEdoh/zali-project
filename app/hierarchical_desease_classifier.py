from typing import Tuple, Dict
from PIL import Image
from constants import PLANT_DISEASES, APPLE_CLASSIFIER_PATH, CORN_CLASSIFIER_PATH, PEPPER_CLASSIFIER_PATH, POTATO_CLASSIFIER_PATH, TOMATO_CLASSIFIER_PATH
from classifiers.plant_classifier import get_classifier, PlantClassifier
from classifiers import get_potato_classifier, get_apple_classifier, get_corn_classifier, get_pepper_classifier, get_tomato_classifier
from classifiers import PotatoDeseaseClassifier, AppleDeseaseClassifier, TomatoDeseaseClassifier, CornDeseaseClassifier, PepperDeseaseClassifier

class HierarchicalPlantDiseaseDetector:
    """
    Complete hierarchical pipeline for plant disease detection.
    
    Pipeline:
    1. Load and preprocess image
    2. Classify plant species using stage 1 model
    3. Load appropriate disease classifier
    4. Classify disease
    5. Return results with confidence scores
    """
    
    def __init__(
        self, 
        plant_classifier_path: str,
        disease_models_path: str,
        img_size: Tuple[int, int] = (224, 224),
        verbose: bool = True
    ):
        """
        Initialize the hierarchical detector.
        
        Args:
            plant_classifier_path: Path to the stage 1 plant classifier model
            disease_models_path: Directory containing disease classifier models
            img_size: Input image size (height, width)
            verbose: Whether to print loading messages
        """
        self.plant_classifier : PlantClassifier = get_classifier(plant_classifier_path)

        self.apple_classifier : AppleDeseaseClassifier = get_apple_classifier(APPLE_CLASSIFIER_PATH)
        self.corn_classifier : CornDeseaseClassifier = get_corn_classifier(CORN_CLASSIFIER_PATH)
        self.pepper_classifier : PepperDeseaseClassifier = get_pepper_classifier(PEPPER_CLASSIFIER_PATH)
        self.tomato_classifier : TomatoDeseaseClassifier = get_tomato_classifier(TOMATO_CLASSIFIER_PATH)
        self.potato_classifier : PotatoDeseaseClassifier = get_potato_classifier(POTATO_CLASSIFIER_PATH)
        
        self.img_size = img_size
        self.verbose = verbose

        if self.verbose:
            print("Loading plant classifier...")

        if self.verbose:
            print("✓ Hierarchical detector initialized")
            print(f"  - Image size: {self.img_size}")
    
    def predict_desease(self, image: Image) -> Dict[str, float]:
        prediction_result = self.plant_classifier.predict_img(image)
        predicted_plant_name = self.get_max_prob_name(prediction_result)

        if predicted_plant_name == "Apple":
            desease_results = self.apple_classifier.predict_img(image)
        elif predicted_plant_name == "Tomato":
            desease_results = self.tomato_classifier.predict_img(image)
        elif predicted_plant_name == "Potato":
            desease_results = self.potato_classifier.predict_img(image)
        elif predicted_plant_name == "Corn_(maize)":
            desease_results = self.corn_classifier.predict_img(image)
        elif predicted_plant_name == "Pepper":
            desease_results = self.pepper_classifier.predict_img(image)
        else:
            desease_results = {}
        return desease_results
        

    def get_max_prob_name(self, prediction_result: Dict[str, float]):
        predicted_plant_name = ""
        max_prob = 0.0
        for plant_name, prob in prediction_result.items():
            if max_prob < prob:
                max_prob = prob
                predicted_plant_name = plant_name
        return predicted_plant_name