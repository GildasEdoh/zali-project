import torch
import torch.nn as nn
from typing import Dict
from torchvision import models, transforms
from PIL import Image
import sys
import matplotlib.pyplot as plt
from typing import Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration - makes hyperparameter tuning easier"""
    
    # Paths
    CSV_PATH = '/kaggle/working/plant_classification_dataset.csv'  # Your CSV file
    MODEL_SAVE_PATH = '/kaggle/working/best_tomato_desease_classifier.pth'
    
    # Model
    MODEL_NAME = 'mobilenet_v2'
    NUM_CLASSES = 10
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 32          # Balanced for laptop GPU (4-8GB VRAM)
    NUM_EPOCHS = 25          # Usually converges by epoch 15-20
    LEARNING_RATE = 0.001    # Conservative for transfer learning
    WEIGHT_DECAY = 1e-4      # L2 regularization to prevent overfitting
    
    # Data split
    VAL_SPLIT = 0.2          # 80% train, 20% validation
    RANDOM_SEED = 42
    
    # Hardware
    NUM_WORKERS = 4          # Parallel data loading (adjust based on CPU cores)
    
    # Class names (must match your CSV target_class values)
    CLASSES = [
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
    ]


class TomatoDeseaseClassifier:
    """
    Wrapper class for easy inference
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize classifier
        
        Args:
            model_path: Path to saved model checkpoint
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on: {self.device}")
        
        # Load checkpoint
        # checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load checkpoint with weights_only=False (PyTorch 2.6+ fix)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        
        # Extract class mapping
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        print(f"Classes: {list(self.class_to_idx.keys())}")
        
        # Create model architecture
        self.model = models.mobilenet_v2(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully!")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Define preprocessing (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_from_path(self, image_path, top_k=3):
        """
        Predict plant type for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions
            
        Returns:
            predictions: List of (class_name, probability) tuples
        """
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return []
        
        # Transform
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
        input_batch = input_batch.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top K predictions
        top_prob, top_idx = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_prob, top_idx):
            class_name = self.idx_to_class[idx.item()]
            predictions.append((class_name, prob.item()))
        
        return predictions
    

    def predict_img(self, image: Image, top_k=3) -> Dict[str, float]:
        # Transform
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
        input_batch = input_batch.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top K predictions
        top_prob, top_idx = torch.topk(probabilities, top_k)
        
        predictions = {}
        for prob, idx in zip(top_prob, top_idx):
            class_name = self.idx_to_class[idx.item()]
            predictions[class_name] = prob.item()
        
        return predictions
    
    def predict_and_display(self, image_path):
        """
        Predict and print results in a user-friendly format
        """
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {image_path}")
        print(f"{'='*60}")
        
        predictions = self.predict(image_path, top_k=self.num_classes)
        
        if not predictions:
            print("Failed to make prediction")
            return
        
        print(f"\nPredictions:")
        for i, (class_name, prob) in enumerate(predictions, 1):
            bar = '█' * int(prob * 50)  # Visual bar
            print(f"{i}. {class_name:20s} {prob*100:5.2f}% {bar}")
        
        # Best prediction
        best_class, best_prob = predictions[0]
        print(f"\n→ Predicted Plant Type: {best_class}")
        print(f"→ Confidence: {best_prob*100:.2f}%")

        img = Image.open(image_path)
        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return predictions

_tomato_desease_classifier : Optional[TomatoDeseaseClassifier] = None

def get_tomato_classifier(model_path) -> TomatoDeseaseClassifier:
    global _tomato_desease_classifier
    if _tomato_desease_classifier is None:
        _tomato_desease_classifier = TomatoDeseaseClassifier(model_path)
    return _tomato_desease_classifier

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    
    # Usage: python inference.py <image_path>
    
    # if len(sys.argv) < 2:
    #     print("Usage: python inference.py <image_path>")
    #     print("Example: python inference.py test_images/tomato_sample.jpg")
    #     sys.exit(1)
    
    image_path = "/Users/v9/Documents/Documents personnels EDOH Yao Gildas/agriVision/app/images/005318c8-a5fa-4420-843b-23bdda7322c2___RS_NLB 3853 copy.jpg"
    model_path = '/Users/v9/Documents/Documents personnels EDOH Yao Gildas/agriVision/app/models/tomato_best_desease_classifier.pth'
    
    # Create classifier
    classifier = TomatoDeseaseClassifier(model_path)
    
    # Predict
    # predictions = classifier.predict_and_display(image_path)
    predictions = classifier.predict_from_path(image_path, top_k=classifier.num_classes)
    print(predictions)