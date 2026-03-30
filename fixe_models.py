import torch



# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration - makes hyperparameter tuning easier"""
    
    # Paths
    CSV_PATH = '/kaggle/working/plant_classification_dataset.csv'  # Your CSV file
    MODEL_SAVE_PATH = '/kaggle/working/best_plant_classifier.pth'
    
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


model_path = "/Users/v9/Documents/Documents personnels EDOH Yao Gildas/agriVision/app/models/tomato_best_desease_classifier.pth"
device = "cpu"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
# Sauvegarder version propre sans Config
clean_checkpoint = {
    'model_state_dict': checkpoint['model_state_dict'],
    'class_to_idx': checkpoint['class_to_idx'],
    'val_acc': checkpoint.get('val_acc', None)
}

torch.save(clean_checkpoint, "/Users/v9/Documents/Documents personnels EDOH Yao Gildas/agriVision/app/models-cleaned/tomato_best_plant_classifier_clean.pth")