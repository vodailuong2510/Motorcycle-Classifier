"""
Main Script for Motorcycle Classifier

This script runs the complete pipeline for motorcycle classification.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.models import get_model
from src.trainer import ModelTrainer
from src.utils import (
    create_directory_structure, 
    save_training_results, 
    generate_classification_report,
    create_experiment_log,
    get_model_summary,
    calculate_model_size
)


def main():
    """Main function to run the motorcycle classification pipeline."""
    
    parser = argparse.ArgumentParser(description='Motorcycle Classifier Training')
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to data directory')
    parser.add_argument('--model_type', type=str, default='vgg16',
                       choices=['simple_cnn', 'vgg16', 'resnet50', 'efficientnet'],
                       help='Type of model to train')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (width height)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default='motorcycle_classifier',
                       help='Name for the experiment')
    
    args = parser.parse_args()
    
    print("=== Motorcycle Classifier Training Pipeline ===")
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Experiment name: {args.experiment_name}")
    print()
    
    # Create directory structure
    print("Creating directory structure...")
    create_directory_structure()
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = DataLoader(
        data_path=args.data_path,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )
    
    # Load data
    print("Loading data...")
    try:
        train_data, val_data, test_data, class_names = data_loader.load_data()
        print(f"Found {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data is organized in the following structure:")
        print("data/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("├── validation/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("└── test/")
        print("    ├── class1/")
        print("    ├── class2/")
        print("    └── ...")
        return
    
    # Create data generators
    print("Creating data generators...")
    train_datagen, val_datagen, test_datagen = data_loader.create_data_generators(
        train_data, val_data, test_data
    )
    
    # Initialize model
    print(f"Initializing {args.model_type} model...")
    model = get_model(
        model_type=args.model_type,
        num_classes=len(class_names),
        img_size=tuple(args.img_size)
    )
    
    # Get model summary
    print("Model Summary:")
    get_model_summary(model.get_model())
    calculate_model_size(model.get_model())
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = ModelTrainer(
        model=model.get_model(),
        model_name=args.experiment_name
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_generator=train_datagen.flow_from_directory(
            os.path.join(args.data_path, 'train'),
            target_size=tuple(args.img_size),
            batch_size=args.batch_size,
            class_mode='categorical'
        ),
        validation_generator=val_datagen.flow_from_directory(
            os.path.join(args.data_path, 'validation'),
            target_size=tuple(args.img_size),
            batch_size=args.batch_size,
            class_mode='categorical'
        ),
        epochs=args.epochs
    )
    
    # Plot training history
    print("Plotting training history...")
    trainer.plot_training_history()
    
    # Evaluate model
    print("Evaluating model...")
    test_generator = test_datagen.flow_from_directory(
        os.path.join(args.data_path, 'test'),
        target_size=tuple(args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    evaluation_results = trainer.evaluate(test_generator)
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    print("Making predictions...")
    predictions, true_labels = trainer.predict(test_generator)
    predicted_labels = predictions.argmax(axis=1)
    
    # Generate classification report
    print("Generating classification report...")
    report = generate_classification_report(
        true_labels, 
        predicted_labels, 
        class_names,
        save_path=f"results/{args.experiment_name}_classification_report.json"
    )
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    trainer.plot_confusion_matrix(
        true_labels, 
        predicted_labels, 
        class_names,
        save_path=f"results/{args.experiment_name}_confusion_matrix.png"
    )
    
    # Save training results
    print("Saving training results...")
    save_training_results(
        history, 
        args.experiment_name,
        save_path="results"
    )
    
    # Save model
    print("Saving model...")
    trainer.save_model(save_path="models")
    
    # Create experiment log
    print("Creating experiment log...")
    config = {
        'data_path': args.data_path,
        'model_type': args.model_type,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    results = {
        'final_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'final_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'test_accuracy': evaluation_results.get('accuracy', 0),
        'test_loss': evaluation_results.get('loss', 0),
        'training_time': str(trainer.training_time) if trainer.training_time else None
    }
    
    create_experiment_log(
        args.experiment_name,
        config,
        results,
        save_path="logs"
    )
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: models/{args.experiment_name}.h5")
    print(f"Results saved to: results/")
    print(f"Logs saved to: logs/")
    print(f"Final test accuracy: {evaluation_results.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    main() 