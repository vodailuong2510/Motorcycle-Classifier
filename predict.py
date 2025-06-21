"""
Prediction Script for Motorcycle Classifier

This script loads a trained model and makes predictions on new images.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import load_image, get_image_info


def load_trained_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tensorflow.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_single_image(model, image_path, class_names, img_size=(224, 224)):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the image
        class_names (list): List of class names
        img_size (tuple): Input image size
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    # Load and preprocess image
    image = load_image(image_path, img_size)
    if image is None:
        return None, None, None
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def predict_batch(model, image_paths, class_names, img_size=(224, 224)):
    """
    Make predictions on a batch of images.
    
    Args:
        model: Trained Keras model
        image_paths (list): List of image paths
        class_names (list): List of class names
        img_size (tuple): Input image size
        
    Returns:
        list: List of prediction results
    """
    results = []
    
    for image_path in image_paths:
        predicted_class, confidence, all_probs = predict_single_image(
            model, image_path, class_names, img_size
        )
        
        if predicted_class is not None:
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': dict(zip(class_names, all_probs))
            }
            results.append(result)
        else:
            print(f"Failed to process image: {image_path}")
    
    return results


def print_prediction_results(results, show_probabilities=False):
    """
    Print prediction results in a formatted way.
    
    Args:
        results (list): List of prediction results
        show_probabilities (bool): Whether to show all class probabilities
    """
    print("\n=== Prediction Results ===")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Image: {os.path.basename(result['image_path'])}")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        if show_probabilities:
            print("   All Class Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                print(f"     {class_name}: {prob:.4f} ({prob*100:.2f}%)")


def main():
    """Main function for making predictions."""
    
    parser = argparse.ArgumentParser(description='Motorcycle Classifier Prediction')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.h5 file)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='List of class names (if not provided, will try to infer from model)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (width height)')
    parser.add_argument('--show_probabilities', action='store_true',
                       help='Show probabilities for all classes')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results as JSON')
    
    args = parser.parse_args()
    
    print("=== Motorcycle Classifier Prediction ===")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image_path}")
    print(f"Image size: {args.img_size}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_trained_model(args.model_path)
    if model is None:
        return
    
    # Get class names
    if args.class_names is None:
        # Try to infer from model output layer
        try:
            num_classes = model.output_shape[-1]
            args.class_names = [f"class_{i}" for i in range(num_classes)]
            print(f"Inferred {num_classes} classes from model")
        except:
            print("Could not infer class names from model. Please provide --class_names")
            return
    
    print(f"Class names: {args.class_names}")
    print()
    
    # Check if image_path is a file or directory
    if os.path.isfile(args.image_path):
        # Single image prediction
        print("Making prediction on single image...")
        predicted_class, confidence, all_probs = predict_single_image(
            model, args.image_path, args.class_names, tuple(args.img_size)
        )
        
        if predicted_class is not None:
            result = {
                'image_path': args.image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': dict(zip(args.class_names, all_probs))
            }
            
            print_prediction_results([result], args.show_probabilities)
            
            # Save results if requested
            if args.output_file:
                import json
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=4)
                print(f"\nResults saved to {args.output_file}")
        else:
            print(f"Failed to process image: {args.image_path}")
    
    elif os.path.isdir(args.image_path):
        # Directory of images
        print("Making predictions on directory of images...")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for file in os.listdir(args.image_path):
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(args.image_path, file))
        
        if not image_paths:
            print(f"No image files found in {args.image_path}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Make predictions
        results = predict_batch(
            model, image_paths, args.class_names, tuple(args.img_size)
        )
        
        if results:
            print_prediction_results(results, args.show_probabilities)
            
            # Save results if requested
            if args.output_file:
                import json
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"\nResults saved to {args.output_file}")
        else:
            print("No images were successfully processed")
    
    else:
        print(f"Error: {args.image_path} is not a valid file or directory")


if __name__ == "__main__":
    main() 