"""
Hierarchical Plant Disease Detection - Inference Pipeline
===========================================================

Production-ready inference script for hierarchical plant disease detection.
Can be used standalone or integrated into web services, APIs, or applications.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --batch path/to/images/ --output results.csv

Author: Plant Disease Detection System v2.0
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


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
        self.img_size = img_size
        self.disease_models_path = Path(disease_models_path)
        self.verbose = verbose
        
        # Load plant classifier
        if self.verbose:
            print("Loading plant classifier...")
        self.plant_classifier = keras.models.load_model(plant_classifier_path)
        
        # Plant classes (should match stage 1 model output order)
        self.plant_classes = ['Apple', 'Corn', 'Pepper', 'Potato', 'Tomato']
        
        # Cache for disease models (lazy loading)
        self.disease_models = {}
        self.disease_class_indices = {}
        
        # Load system metadata if available
        metadata_path = self.disease_models_path / 'system_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
        
        if self.verbose:
            print("✓ Hierarchical detector initialized")
            print(f"  - Image size: {self.img_size}")
            print(f"  - Plant classes: {', '.join(self.plant_classes)}")
    
    def load_disease_model(self, plant_name: str) -> Tuple:
        """
        Lazy loads disease model for specific plant.
        
        Args:
            plant_name: Name of the plant species
            
        Returns:
            Tuple of (model, class_indices)
        """
        if plant_name not in self.disease_models:
            model_path = self.disease_models_path / f'{plant_name}_disease_classifier.h5'
            indices_path = self.disease_models_path / f'{plant_name}_class_indices.json'
            
            if not model_path.exists():
                raise FileNotFoundError(f"Disease model not found: {model_path}")
            
            if self.verbose:
                print(f"Loading {plant_name} disease classifier...")
            
            self.disease_models[plant_name] = keras.models.load_model(str(model_path))
            
            with open(indices_path, 'r') as f:
                self.disease_class_indices[plant_name] = json.load(f)
        
        return self.disease_models[plant_name], self.disease_class_indices[plant_name]
    
    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Loads and preprocesses an image for prediction.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image array
        """
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.img_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path: Union[str, Path], top_k: int = 3) -> Dict:
        """
        Performs hierarchical prediction on an image.
        
        Args:
            image_path: Path to the input image
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary containing:
            - plant: Detected plant species
            - plant_confidence: Confidence score for plant prediction
            - disease: Detected disease
            - disease_confidence: Confidence score for disease prediction
            - full_diagnosis: Complete diagnosis string
            - top_predictions: List of top k predictions
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Stage 1: Classify plant species
        plant_predictions = self.plant_classifier.predict(img_array, verbose=0)[0]
        plant_idx = np.argmax(plant_predictions)
        plant_name = self.plant_classes[plant_idx]
        plant_confidence = float(plant_predictions[plant_idx])
        
        # Stage 2: Classify disease
        disease_model, class_indices = self.load_disease_model(plant_name)
        disease_predictions = disease_model.predict(img_array, verbose=0)[0]
        
        # Get reverse mapping (index -> disease name)
        idx_to_disease = {v: k for k, v in class_indices.items()}
        
        # Get top k predictions
        top_indices = np.argsort(disease_predictions)[-top_k:][::-1]
        top_predictions = [
            {
                'disease': idx_to_disease[idx],
                'confidence': float(disease_predictions[idx])
            }
            for idx in top_indices
        ]
        
        # Main prediction
        disease_idx = top_indices[0]
        disease_name = idx_to_disease[disease_idx]
        disease_confidence = float(disease_predictions[disease_idx])
        
        return {
            'plant': plant_name,
            'plant_confidence': plant_confidence,
            'disease': disease_name,
            'disease_confidence': disease_confidence,
            'full_diagnosis': f'{plant_name}___{disease_name}',
            'top_predictions': top_predictions,
            'image_path': str(image_path)
        }
    
    def predict_batch(
        self, 
        image_paths: List[Union[str, Path]], 
        progress: bool = True
    ) -> List[Dict]:
        """
        Performs hierarchical prediction on multiple images.
        
        Args:
            image_paths: List of paths to input images
            progress: Whether to show progress
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(image_paths)
        
        for i, img_path in enumerate(image_paths, 1):
            if progress and self.verbose:
                print(f"Processing {i}/{total}: {Path(img_path).name}", end='\r')
            
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        if progress and self.verbose:
            print(f"\n✓ Processed {total} images")
        
        return results
    
    def predict_directory(
        self, 
        directory: Union[str, Path], 
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    ) -> List[Dict]:
        """
        Performs prediction on all images in a directory.
        
        Args:
            directory: Path to directory containing images
            extensions: List of valid image extensions
            
        Returns:
            List of prediction results
        """
        directory = Path(directory)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
        
        if self.verbose:
            print(f"Found {len(image_paths)} images in {directory}")
        
        return self.predict_batch(image_paths)
    
    def get_info(self) -> Dict:
        """Returns information about the detector configuration."""
        return {
            'plant_classes': self.plant_classes,
            'image_size': self.img_size,
            'loaded_disease_models': list(self.disease_models.keys()),
            'metadata': self.metadata
        }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Hierarchical Plant Disease Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python inference.py --image path/to/image.jpg
  
  # Batch prediction on directory
  python inference.py --batch path/to/images/ --output results.csv
  
  # With custom models path
  python inference.py --image test.jpg --models /path/to/models
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for prediction'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Path to directory containing images for batch prediction'
    )
    
    parser.add_argument(
        '--plant-model',
        type=str,
        default='plant_classifier.h5',
        help='Path to plant classifier model (default: plant_classifier.h5)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='disease_models',
        help='Path to directory containing disease models (default: disease_models)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path for batch predictions'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show (default: 3)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.error("Either --image or --batch must be specified")
    
    # Initialize detector
    print("=" * 70)
    print("HIERARCHICAL PLANT DISEASE DETECTION SYSTEM")
    print("=" * 70)
    
    detector = HierarchicalPlantDiseaseDetector(
        plant_classifier_path=args.plant_model,
        disease_models_path=args.models,
        verbose=not args.quiet
    )
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting: {args.image}")
        print("-" * 70)
        
        result = detector.predict(args.image, top_k=args.top_k)
        
        print(f"\n{'Plant:':<20} {result['plant']}")
        print(f"{'Plant Confidence:':<20} {result['plant_confidence']:.2%}")
        print(f"{'Disease:':<20} {result['disease']}")
        print(f"{'Disease Confidence:':<20} {result['disease_confidence']:.2%}")
        print(f"{'Full Diagnosis:':<20} {result['full_diagnosis']}")
        
        print(f"\nTop {args.top_k} Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['disease']:<40} {pred['confidence']:.2%}")
    
    # Batch prediction
    if args.batch:
        print(f"\nBatch prediction on: {args.batch}")
        print("-" * 70)
        
        results = detector.predict_directory(args.batch)
        
        # Create DataFrame
        df_results = pd.DataFrame([
            {
                'Image': Path(r['image_path']).name,
                'Plant': r.get('plant', 'ERROR'),
                'Plant_Confidence': r.get('plant_confidence', 0),
                'Disease': r.get('disease', 'ERROR'),
                'Disease_Confidence': r.get('disease_confidence', 0),
                'Full_Diagnosis': r.get('full_diagnosis', 'ERROR'),
                'Error': r.get('error', '')
            }
            for r in results
        ])
        
        print(f"\nResults Preview:")
        print(df_results[['Image', 'Plant', 'Disease', 'Disease_Confidence']].head(10).to_string(index=False))
        
        # Save to CSV if output path specified
        if args.output:
            df_results.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
        else:
            print(f"\n✓ Processed {len(results)} images")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
