"""
Utility Functions Module

This module contains utility functions for the motorcycle classifier project.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import classification_report
import pandas as pd
from datetime import datetime


def create_directory_structure(base_path="data"):
    """
    Create the standard directory structure for the project.
    
    Args:
        base_path (str): Base path for data directory
    """
    directories = [
        base_path,
        os.path.join(base_path, "train"),
        os.path.join(base_path, "validation"),
        os.path.join(base_path, "test"),
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image.
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image_array, save_path, denormalize=True):
    """
    Save an image array to file.
    
    Args:
        image_array (numpy.ndarray): Image array to save
        save_path (str): Path to save the image
        denormalize (bool): Whether to denormalize the image (multiply by 255)
    """
    try:
        if denormalize:
            image_array = (image_array * 255).astype(np.uint8)
        
        img = Image.fromarray(image_array)
        img.save(save_path)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")


def plot_sample_images(data_generator, num_samples=8, class_names=None):
    """
    Plot sample images from a data generator.
    
    Args:
        data_generator: Data generator
        num_samples (int): Number of samples to plot
        class_names (list): List of class names
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        batch = next(data_generator)
        if isinstance(batch, tuple):
            image = batch[0][0]
            label = batch[1][0]
        else:
            image = batch[0]
            label = np.zeros(image.shape[0])
        
        axes[i].imshow(image)
        if class_names:
            class_idx = np.argmax(label)
            axes[i].set_title(class_names[class_idx])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_training_results(history, model_name, save_path="results"):
    """
    Save training results to files.
    
    Args:
        history: Training history from Keras
        model_name (str): Name of the model
        save_path (str): Directory to save results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save training history as JSON
    history_dict = {}
    for key in history.history.keys():
        history_dict[key] = [float(x) for x in history.history[key]]
    
    history_path = os.path.join(save_path, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Save training curves plot
    plot_path = os.path.join(save_path, f"{model_name}_training_curves.png")
    plot_training_curves(history, save_path=plot_path)
    
    print(f"Training results saved to {save_path}")


def plot_training_curves(history, save_path=None):
    """
    Plot training curves.
    
    Args:
        history: Training history from Keras
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Print report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Classification report saved to {save_path}")
    
    return report


def create_experiment_log(experiment_name, config, results, save_path="logs"):
    """
    Create an experiment log entry.
    
    Args:
        experiment_name (str): Name of the experiment
        config (dict): Configuration parameters
        results (dict): Results of the experiment
        save_path (str): Directory to save logs
    """
    os.makedirs(save_path, exist_ok=True)
    
    log_entry = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results
    }
    
    log_file = os.path.join(save_path, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=4)
    
    print(f"Experiment log saved to {log_file}")


def get_model_summary(model, save_path=None):
    """
    Get and optionally save model summary.
    
    Args:
        model: Keras model
        save_path (str): Path to save the summary
    """
    # Capture model summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_str = '\n'.join(summary_list)
    
    print("Model Summary:")
    print(summary_str)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary_str)
        print(f"Model summary saved to {save_path}")
    
    return summary_str


def calculate_model_size(model, save_path=None):
    """
    Calculate model size in MB.
    
    Args:
        model: Keras model
        save_path (str): Path to save the size info
        
    Returns:
        float: Model size in MB
    """
    param_count = model.count_params()
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    
    size_info = {
        'parameters': param_count,
        'size_mb': model_size_mb,
        'layers': len(model.layers)
    }
    
    print(f"Model Parameters: {param_count:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Number of Layers: {len(model.layers)}")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(size_info, f, indent=4)
        print(f"Model size info saved to {save_path}")
    
    return model_size_mb


def validate_image_file(image_path):
    """
    Validate if a file is a valid image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path):
    """
    Get information about an image.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        dict: Image information
    """
    try:
        with Image.open(image_path) as img:
            info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height
            }
        return info
    except Exception as e:
        print(f"Error getting image info for {image_path}: {e}")
        return None 