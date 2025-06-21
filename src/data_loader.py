"""
Data Loading and Preprocessing Module

This module handles loading, preprocessing, and data augmentation for motorcycle images.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    """Class for loading and preprocessing motorcycle image data."""
    
    def __init__(self, data_path, img_size=(224, 224), batch_size=32):
        """
        Initialize DataLoader.
        
        Args:
            data_path (str): Path to the data directory
            img_size (tuple): Target image size (width, height)
            batch_size (int): Batch size for training
        """
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}
        
    def load_data(self):
        """
        Load and organize data from directory structure.
        
        Returns:
            tuple: (train_data, val_data, test_data, class_names)
        """
        # Implementation will be added based on notebook content
        pass
    
    def create_data_generators(self, train_data, val_data, test_data):
        """
        Create data generators with augmentation.
        
        Args:
            train_data: Training data
            val_data: Validation data  
            test_data: Test data
            
        Returns:
            tuple: (train_gen, val_gen, test_gen)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, val_datagen, test_datagen
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return img_array 