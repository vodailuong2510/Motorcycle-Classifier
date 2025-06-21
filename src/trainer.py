"""
Model Training Module

This module handles model training, validation, and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime


class ModelTrainer:
    """Class for training and evaluating motorcycle classification models."""
    
    def __init__(self, model, model_name="motorcycle_classifier"):
        """
        Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            model_name (str): Name for saving the model
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        self.training_time = None
        
    def setup_callbacks(self, checkpoint_dir="models", patience=10):
        """
        Setup training callbacks.
        
        Args:
            checkpoint_dir (str): Directory to save model checkpoints
            patience (int): Patience for early stopping
            
        Returns:
            list: List of callbacks
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{self.model_name}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, validation_generator, epochs=50, 
              steps_per_epoch=None, validation_steps=None, callbacks=None):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            steps_per_epoch (int): Steps per epoch
            validation_steps (int): Validation steps
            callbacks (list): Training callbacks
            
        Returns:
            dict: Training history
        """
        if callbacks is None:
            callbacks = self.setup_callbacks()
        
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = datetime.now() - start_time
        
        return self.history
    
    def evaluate(self, test_generator, test_steps=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_generator: Test data generator
            test_steps (int): Test steps
            
        Returns:
            dict: Evaluation results
        """
        results = self.model.evaluate(
            test_generator,
            steps=test_steps,
            verbose=1
        )
        
        metrics = dict(zip(self.model.metrics_names, results))
        return metrics
    
    def predict(self, test_generator, test_steps=None):
        """
        Make predictions on test data.
        
        Args:
            test_generator: Test data generator
            test_steps (int): Test steps
            
        Returns:
            tuple: (predictions, true_labels)
        """
        predictions = self.model.predict(
            test_generator,
            steps=test_steps,
            verbose=1
        )
        
        # Get true labels
        true_labels = []
        for i in range(test_steps or len(test_generator)):
            batch = next(test_generator)
            if isinstance(batch, tuple):
                true_labels.extend(np.argmax(batch[1], axis=1))
            else:
                true_labels.extend(np.argmax(batch, axis=1))
        
        return predictions, np.array(true_labels)
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, save_path="models"):
        """
        Save the trained model.
        
        Args:
            save_path (str): Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_path, f"{self.model_name}.h5")
        self.model.save(model_path)
        
        # Save training info
        info = {
            'model_name': self.model_name,
            'training_time': str(self.training_time) if self.training_time else None,
            'final_accuracy': self.history.history['accuracy'][-1] if self.history else None,
            'final_val_accuracy': self.history.history['val_accuracy'][-1] if self.history else None,
            'epochs_trained': len(self.history.history['accuracy']) if self.history else 0
        }
        
        info_path = os.path.join(save_path, f"{self.model_name}_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Model saved to {model_path}")
        print(f"Training info saved to {info_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}") 