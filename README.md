# Hierarchical Plant Disease Detection System

A production-ready deep learning system for hierarchical plant disease detection using MobileNetV3 with transfer learning.

## 🌟 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   STAGE 1: Plant Classifier │
        │   (Pre-trained Model)       │
        │   - Apple                   │
        │   - Tomato                  │
        │   - Potato                  │
        │   - Corn                    │
        │   - Pepper                  │
        └─────────────┬───────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  Route to Specific Disease  │
        │      Classifier             │
        └─────────────┬───────────────┘
                      │
          ┌───────────┴───────────┬───────────┬───────────┬───────────┐
          ▼                       ▼           ▼           ▼           ▼
    ┌─────────┐            ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Apple   │            │ Tomato  │  │ Potato  │  │  Corn   │  │ Pepper  │
    │ Disease │            │ Disease │  │ Disease │  │ Disease │  │ Disease │
    │Classifier│           │Classifier│ │Classifier│ │Classifier│ │Classifier│
    │(4 class)│            │(10 class)│ │(3 class)│ │(4 class)│ │(2 class)│
    └────┬────┘            └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
         │                      │            │            │            │
         └──────────────────────┴────────────┴────────────┴────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   FINAL PREDICTION      │
                              │   Plant + Disease       │
                              │   + Confidence Scores   │
                              └─────────────────────────┘
```

## 📋 Disease Classes by Plant

### Apple (4 classes)
- Black_rot
- Cedar_apple_rust
- Apple_scab
- healthy

### Tomato (10 classes)
- Bacterial_spot
- Early_blight
- Late_blight
- Leaf_Mold
- Septoria_leaf_spot
- Spider_mites Two-spotted_spider_mite
- Target_Spot
- Tomato_Yellow_Leaf_Curl_Virus
- Tomato_mosaic_virus
- healthy

### Potato (3 classes)
- Early_blight
- Late_blight
- healthy

### Corn (4 classes)
- Cercospora_leaf_spot Gray_leaf_spot
- Common_rust
- Northern_Leaf_Blight
- healthy

### Pepper (2 classes)
- Bacterial_spot
- healthy

## 🏗️ Model Architecture

### Base Architecture: MobileNetV3Large
- **Pretrained on**: ImageNet
- **Input size**: 224x224x3
- **Parameters**: ~5.4M (per model)
- **Optimization**: Mixed precision training (FP16)

### Classifier Head
```
Input (224x224x3)
    ↓
MobileNetV3Large (frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(256, relu)
    ↓
Dropout(0.3)
    ↓
Dense(num_classes, softmax)
```

### Training Strategy
1. **Phase 1**: Train classifier head (base frozen)
   - Epochs: 15
   - Learning rate: 0.001
   
2. **Phase 2**: Fine-tune top 30 layers
   - Epochs: 15
   - Learning rate: 0.0001

## 📊 Performance Metrics

| Plant   | Classes | Val Accuracy | Parameters |
|---------|---------|--------------|------------|
| Apple   | 4       | ~98%         | 5.4M       |
| Tomato  | 10      | ~95%         | 5.4M       |
| Potato  | 3       | ~99%         | 5.4M       |
| Corn    | 4       | ~97%         | 5.4M       |
| Pepper  | 2       | ~99%         | 5.4M       |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Training on Kaggle

1. Upload the notebook `hierarchical_plant_disease_detection.ipynb` to Kaggle
2. Add PlantVillage dataset
3. Add your Stage 1 plant classifier model
4. Enable GPU accelerator
5. Run all cells

### Inference

#### Single Image
```python
from inference import HierarchicalPlantDiseaseDetector

detector = HierarchicalPlantDiseaseDetector(
    plant_classifier_path='plant_classifier.h5',
    disease_models_path='disease_models/'
)

result = detector.predict('path/to/image.jpg')
print(f"Plant: {result['plant']}")
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['disease_confidence']:.2%}")
```

#### Batch Processing
```python
results = detector.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for result in results:
    print(f"{result['image_path']}: {result['full_diagnosis']}")
```

#### Command Line
```bash
# Single image
python inference.py --image test.jpg

# Batch processing
python inference.py --batch ./test_images/ --output results.csv
```

## 📁 File Structure

```
.
├── hierarchical_plant_disease_detection.ipynb  # Main training notebook
├── inference.py                                 # Standalone inference script
├── README.md                                    # This file
├── disease_models/                              # Trained models directory
│   ├── Apple_disease_classifier.h5
│   ├── Apple_class_indices.json
│   ├── Tomato_disease_classifier.h5
│   ├── Tomato_class_indices.json
│   ├── Potato_disease_classifier.h5
│   ├── Potato_class_indices.json
│   ├── Corn_disease_classifier.h5
│   ├── Corn_class_indices.json
│   ├── Pepper_disease_classifier.h5
│   ├── Pepper_class_indices.json
│   ├── system_metadata.json
│   └── model_performance_summary.csv
└── reorganized_dataset/                         # Training data
    ├── Apple/
    │   ├── train/
    │   └── val/
    ├── Tomato/
    ├── Potato/
    ├── Corn/
    └── Pepper/
```

## 🔧 Configuration

### Training Configuration
```python
class Config:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
```

### Data Augmentation
- Random rotation: ±15°
- Width/height shift: ±10%
- Zoom: ±20%
- Horizontal flip: True
- Shear: 0.1
- Brightness: [0.8, 1.2]

## 🎯 Key Features

### 1. Hierarchical Architecture
- **Memory Efficient**: Only loads relevant disease model
- **Scalable**: Easy to add new plants or diseases
- **Fast Inference**: Lazy loading of models

### 2. Transfer Learning
- Leverages ImageNet pretrained weights
- Fast convergence
- High accuracy with limited data

### 3. Mixed Precision Training
- 2x faster training on modern GPUs
- Reduced memory footprint
- Maintained accuracy

### 4. Production Ready
- Comprehensive error handling
- Batch processing support
- CSV export functionality
- Command-line interface

## 📈 Training Best Practices

### Data Organization
```
PlantVillage/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Tomato___Bacterial_spot/
└── ...
```

### Callbacks Used
1. **EarlyStopping**: Prevents overfitting
   - Monitor: val_accuracy
   - Patience: 5 epochs

2. **ReduceLROnPlateau**: Adaptive learning rate
   - Monitor: val_loss
   - Factor: 0.5
   - Patience: 3 epochs

3. **ModelCheckpoint**: Saves best model
   - Monitor: val_accuracy
   - Save best only

## 🔄 Deployment Options

### Option 1: Python API
```python
# Flask example
from flask import Flask, request, jsonify
from inference import HierarchicalPlantDiseaseDetector

app = Flask(__name__)
detector = HierarchicalPlantDiseaseDetector(
    plant_classifier_path='models/plant_classifier.h5',
    disease_models_path='models/disease_models/'
)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp.jpg')
    result = detector.predict('temp.jpg')
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 2: TensorFlow Lite (Mobile)
```python
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Option 3: ONNX (Cross-platform)
```python
import tf2onnx

# Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

### Option 4: Docker Container
```dockerfile
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inference.py .
COPY disease_models/ disease_models/
COPY plant_classifier.h5 .

EXPOSE 5000
CMD ["python", "api.py"]
```

## 🧪 Testing

### Unit Tests
```python
import unittest
from inference import HierarchicalPlantDiseaseDetector

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HierarchicalPlantDiseaseDetector(
            plant_classifier_path='plant_classifier.h5',
            disease_models_path='disease_models/'
        )
    
    def test_single_prediction(self):
        result = self.detector.predict('test_image.jpg')
        self.assertIn('plant', result)
        self.assertIn('disease', result)
        self.assertIsInstance(result['disease_confidence'], float)
```

### Performance Benchmarks
```python
import time

# Measure inference time
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
start = time.time()
results = detector.predict_batch(images)
elapsed = time.time() - start

print(f"Average inference time: {elapsed/len(images):.3f}s per image")
```

## 📝 Output Format

### Single Prediction
```json
{
  "plant": "Tomato",
  "plant_confidence": 0.9876,
  "disease": "Early_blight",
  "disease_confidence": 0.9543,
  "full_diagnosis": "Tomato___Early_blight",
  "top_predictions": [
    {"disease": "Early_blight", "confidence": 0.9543},
    {"disease": "Late_blight", "confidence": 0.0321},
    {"disease": "healthy", "confidence": 0.0087}
  ],
  "image_path": "path/to/image.jpg"
}
```

### Batch Results (CSV)
```csv
Image,Plant,Plant_Confidence,Disease,Disease_Confidence,Full_Diagnosis
img1.jpg,Apple,0.99,Black_rot,0.96,Apple___Black_rot
img2.jpg,Tomato,0.98,Early_blight,0.94,Tomato___Early_blight
img3.jpg,Potato,0.99,healthy,0.97,Potato___healthy
```

## 🎓 Educational Resources

### Understanding the Architecture
1. **Why Hierarchical?**
   - Reduces complexity (5 binary classifiers vs 1 multi-class)
   - Better accuracy per plant
   - More interpretable predictions

2. **Transfer Learning Benefits**
   - Faster training
   - Better generalization
   - Requires less data

3. **MobileNetV3 Advantages**
   - Lightweight (5.4M params)
   - Fast inference
   - Mobile-friendly

## 🐛 Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```python
# Solution: Reduce batch size
config.BATCH_SIZE = 16  # Instead of 32
```

**Issue**: Model not loading
```python
# Solution: Use absolute paths
import os
model_path = os.path.abspath('disease_models/Apple_disease_classifier.h5')
```

**Issue**: Low accuracy on new images
```python
# Solution: Check image preprocessing
# Ensure images are RGB and normalized
img = img.convert('RGB')
img_array = img_array / 255.0
```

## 📊 Monitoring & Logging

### Add TensorBoard
```python
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True
)

model.fit(train_gen, callbacks=[tensorboard, ...])
```

### View Logs
```bash
tensorboard --logdir=./logs
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PlantVillage dataset creators
- TensorFlow/Keras team
- MobileNetV3 authors

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@example.com
- GitHub: @yourusername

## 🔗 Related Resources

- [PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

**Built with ❤️ for sustainable agriculture**
