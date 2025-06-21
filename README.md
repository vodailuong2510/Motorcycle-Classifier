# Motorcycle Classifier

A machine learning project for classifying motorcycle images using deep learning techniques.

## Project Structure

```
Motorcycle-Classifier/
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── models.py          # Neural network model definitions
│   ├── trainer.py         # Model training and evaluation
│   └── utils.py           # Utility functions
├── main.py                # Main training script
├── predict.py             # Prediction script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE.txt           # License information
└── Motorcycle_Classifier.ipynb  # Original Jupyter notebook
```

## Features

- **Multiple Model Architectures**: Support for Simple CNN, VGG16, ResNet50, and EfficientNet
- **Transfer Learning**: Pre-trained models with fine-tuning capabilities
- **Data Augmentation**: Built-in image augmentation for better generalization
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and training curves
- **Experiment Logging**: Automatic logging of experiments and results
- **Easy-to-use Scripts**: Command-line interfaces for training and prediction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Motorcycle-Classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Organization

Organize your data in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── validation/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Usage

### Training a Model

Use the main script to train a model:

```bash
python main.py --data_path data --model_type vgg16 --epochs 50 --batch_size 32
```

#### Command Line Arguments:

- `--data_path`: Path to data directory (default: 'data')
- `--model_type`: Type of model ('simple_cnn', 'vgg16', 'resnet50', 'efficientnet') (default: 'vgg16')
- `--img_size`: Image size as width height (default: 224 224)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--experiment_name`: Name for the experiment (default: 'motorcycle_classifier')

#### Examples:

```bash
# Train with VGG16
python main.py --model_type vgg16 --epochs 100 --batch_size 16

# Train with ResNet50
python main.py --model_type resnet50 --img_size 299 299 --epochs 75

# Train with custom settings
python main.py --model_type efficientnet --epochs 200 --batch_size 8 --learning_rate 0.0001
```

### Making Predictions

Use the prediction script to classify new images:

```bash
python predict.py --model_path models/motorcycle_classifier.h5 --image_path path/to/image.jpg
```

#### Command Line Arguments:

- `--model_path`: Path to the trained model (.h5 file) (required)
- `--image_path`: Path to image or directory of images (required)
- `--class_names`: List of class names (optional, will be inferred from model if not provided)
- `--img_size`: Image size as width height (default: 224 224)
- `--show_probabilities`: Show probabilities for all classes
- `--output_file`: Path to save results as JSON

#### Examples:

```bash
# Predict single image
python predict.py --model_path models/motorcycle_classifier.h5 --image_path test_image.jpg

# Predict with class names
python predict.py --model_path models/motorcycle_classifier.h5 --image_path test_image.jpg --class_names sport touring cruiser

# Predict directory of images
python predict.py --model_path models/motorcycle_classifier.h5 --image_path test_images/ --show_probabilities

# Save results to file
python predict.py --model_path models/motorcycle_classifier.h5 --image_path test_images/ --output_file results.json
```

## Module Documentation

### Data Loader (`src/data_loader.py`)

Handles data loading, preprocessing, and augmentation:

```python
from src.data_loader import DataLoader

# Initialize data loader
data_loader = DataLoader(data_path="data", img_size=(224, 224), batch_size=32)

# Load data
train_data, val_data, test_data, class_names = data_loader.load_data()

# Create data generators
train_datagen, val_datagen, test_datagen = data_loader.create_data_generators(
    train_data, val_data, test_data
)
```

### Models (`src/models.py`)

Contains model architectures and factory function:

```python
from src.models import get_model

# Get a model
model = get_model(
    model_type='vgg16',
    num_classes=3,
    img_size=(224, 224)
)

# Get the compiled model
keras_model = model.get_model()
```

### Trainer (`src/trainer.py`)

Handles model training, evaluation, and visualization:

```python
from src.trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(model=model, model_name="my_model")

# Train model
history = trainer.train(train_generator, validation_generator, epochs=50)

# Evaluate model
results = trainer.evaluate(test_generator)

# Plot training history
trainer.plot_training_history()

# Save model
trainer.save_model()
```

### Utils (`src/utils.py`)

Utility functions for various tasks:

```python
from src.utils import (
    create_directory_structure,
    load_image,
    save_training_results,
    generate_classification_report
)

# Create project directories
create_directory_structure()

# Load and preprocess image
image = load_image("path/to/image.jpg", target_size=(224, 224))

# Save training results
save_training_results(history, "experiment_name")
```

## Output Files

After training, the following files will be created:

- `models/`: Trained model files (.h5)
- `results/`: Training curves, confusion matrices, and classification reports
- `logs/`: Experiment logs with configuration and results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.
