"""
Example Usage Script

This script demonstrates how to use the motorcycle classifier modules.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.models import get_model
from src.trainer import ModelTrainer
from src.utils import (
    create_directory_structure,
    load_image,
    plot_sample_images,
    save_training_results,
    generate_classification_report
)


def example_data_loading():
    """Example of data loading and preprocessing."""
    print("=== Data Loading Example ===")
    
    # Create directory structure
    create_directory_structure()
    
    # Initialize data loader
    data_loader = DataLoader(
        data_path="data",
        img_size=(224, 224),
        batch_size=32
    )
    
    print("Data loader initialized")
    print(f"Image size: {data_loader.img_size}")
    print(f"Batch size: {data_loader.batch_size}")
    
    # Note: You would need actual data in the data/ directory to run this
    # try:
    #     train_data, val_data, test_data, class_names = data_loader.load_data()
    #     print(f"Found classes: {class_names}")
    # except Exception as e:
    #     print(f"Error loading data: {e}")
    
    print()


def example_model_creation():
    """Example of creating different model types."""
    print("=== Model Creation Example ===")
    
    model_types = ['simple_cnn', 'vgg16', 'resnet50', 'efficientnet']
    num_classes = 5
    img_size = (224, 224)
    
    for model_type in model_types:
        print(f"\nCreating {model_type} model...")
        try:
            model = get_model(
                model_type=model_type,
                num_classes=num_classes,
                img_size=img_size
            )
            
            keras_model = model.get_model()
            print(f"✓ {model_type} model created successfully")
            print(f"  Parameters: {keras_model.count_params():,}")
            print(f"  Output shape: {keras_model.output_shape}")
            
        except Exception as e:
            print(f"✗ Error creating {model_type} model: {e}")
    
    print()


def example_image_processing():
    """Example of image processing utilities."""
    print("=== Image Processing Example ===")
    
    # Example image path (you would need a real image)
    example_image_path = "example_image.jpg"
    
    if os.path.exists(example_image_path):
        # Load and preprocess image
        image = load_image(example_image_path, target_size=(224, 224))
        if image is not None:
            print(f"✓ Image loaded successfully")
            print(f"  Shape: {image.shape}")
            print(f"  Data type: {image.dtype}")
            print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
        else:
            print("✗ Failed to load image")
    else:
        print(f"Example image not found: {example_image_path}")
        print("Create an example image to test this functionality")
    
    print()


def example_training_pipeline():
    """Example of complete training pipeline."""
    print("=== Training Pipeline Example ===")
    
    # This is a conceptual example - you would need actual data to run it
    print("Complete training pipeline would include:")
    print("1. Data loading and preprocessing")
    print("2. Model creation")
    print("3. Training with callbacks")
    print("4. Evaluation and visualization")
    print("5. Model saving")
    
    # Example code structure:
    """
    # 1. Initialize components
    data_loader = DataLoader(data_path="data", img_size=(224, 224), batch_size=32)
    model = get_model(model_type='vgg16', num_classes=5, img_size=(224, 224))
    trainer = ModelTrainer(model=model.get_model(), model_name="example_model")
    
    # 2. Load data
    train_data, val_data, test_data, class_names = data_loader.load_data()
    train_datagen, val_datagen, test_datagen = data_loader.create_data_generators(
        train_data, val_data, test_data
    )
    
    # 3. Train model
    history = trainer.train(
        train_generator=train_datagen.flow_from_directory(...),
        validation_generator=val_datagen.flow_from_directory(...),
        epochs=50
    )
    
    # 4. Evaluate and save results
    results = trainer.evaluate(test_generator)
    trainer.plot_training_history()
    trainer.save_model()
    save_training_results(history, "example_model")
    """
    
    print()


def example_prediction():
    """Example of making predictions."""
    print("=== Prediction Example ===")
    
    # This would require a trained model
    model_path = "models/example_model.h5"
    
    if os.path.exists(model_path):
        print(f"✓ Found trained model: {model_path}")
        print("You can use this model for predictions:")
        print("python predict.py --model_path models/example_model.h5 --image_path test_image.jpg")
    else:
        print("No trained model found")
        print("Train a model first using main.py or the training pipeline")
    
    print()


def main():
    """Run all examples."""
    print("Motorcycle Classifier - Example Usage")
    print("=" * 50)
    
    example_data_loading()
    example_model_creation()
    example_image_processing()
    example_training_pipeline()
    example_prediction()
    
    print("=== Summary ===")
    print("To get started:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Organize your data in the data/ directory")
    print("3. Train a model: python main.py --data_path data --model_type vgg16")
    print("4. Make predictions: python predict.py --model_path models/model.h5 --image_path image.jpg")
    print()
    print("For more information, see README.md")


if __name__ == "__main__":
    main() 