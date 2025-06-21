"""
Model Definitions Module

This module contains various neural network architectures for motorcycle classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0


class MotorcycleClassifier:
    """Base class for motorcycle classification models."""
    
    def __init__(self, num_classes, img_size=(224, 224)):
        """
        Initialize the classifier.
        
        Args:
            num_classes (int): Number of motorcycle classes
            img_size (tuple): Input image size (width, height)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def build_model(self):
        """Build the neural network model."""
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Get the compiled model."""
        return self.model


class SimpleCNN(MotorcycleClassifier):
    """Simple Convolutional Neural Network for motorcycle classification."""
    
    def build_model(self):
        """Build a simple CNN architecture."""
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])


class VGG16Classifier(MotorcycleClassifier):
    """VGG16-based classifier with transfer learning."""
    
    def build_model(self):
        """Build VGG16-based model with transfer learning."""
        # Load pre-trained VGG16
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])


class ResNet50Classifier(MotorcycleClassifier):
    """ResNet50-based classifier with transfer learning."""
    
    def build_model(self):
        """Build ResNet50-based model with transfer learning."""
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])


class EfficientNetClassifier(MotorcycleClassifier):
    """EfficientNet-based classifier with transfer learning."""
    
    def build_model(self):
        """Build EfficientNet-based model with transfer learning."""
        # Load pre-trained EfficientNet
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])


def get_model(model_type, num_classes, img_size=(224, 224)):
    """
    Factory function to create different model types.
    
    Args:
        model_type (str): Type of model ('simple_cnn', 'vgg16', 'resnet50', 'efficientnet')
        num_classes (int): Number of classes
        img_size (tuple): Input image size
        
    Returns:
        MotorcycleClassifier: Compiled model instance
    """
    model_map = {
        'simple_cnn': SimpleCNN,
        'vgg16': VGG16Classifier,
        'resnet50': ResNet50Classifier,
        'efficientnet': EfficientNetClassifier
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = model_map[model_type]
    model = model_class(num_classes, img_size)
    model.compile_model()
    
    return model 