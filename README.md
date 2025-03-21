# Content Sentry

A deep learning-based multimodal content classifier for hateful meme detection, built from scratch.

## Overview

Content Sentry is a custom implementation of a multimodal deep learning system that processes both images and text to detect hateful content in memes. The system features:

- Custom neural network implementations in NumPy
- CNN-based image processing
- Text processing with embeddings and convolutions
- Multimodal fusion for final classification
- Multiple architectural variations with different performance characteristics

## Project Structure

- **framework/**: Core neural network components implemented from scratch
  - Layer abstractions (convolutional, pooling, fully connected, etc.)
  - Activation functions (ReLU, Sigmoid, etc.)
  - Loss functions
  - Normalization techniques
  
- **models/**: Higher-level model architectures
  - `image_cnn.py`: CNN architecture for image processing
  - `text_cnn.py`: CNN architecture for text processing
  - `multimodal.py`: Fusion model combining image and text features
  
- **Core scripts**:
  - `simple_cnn.py`: Basic implementation with training and evaluation
  - `advanced_cnn.py`: Enhanced implementation with improved regularization and optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/eparirishit/content-sentry.git
cd content-sentry

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

Run basic training with simpler models:

```bash
python simple_cnn.py
```

### Advanced Training

Run advanced training with improved models and training techniques:

```bash
python advanced_cnn.py
```

## Dataset

This project uses the "Hateful Memes" dataset, which contains memes labeled as either hateful or non-hateful. The dataset is automatically downloaded using the Hugging Face datasets library.

## Model Architectures

- **ImageCNN**: Processes images through multiple convolutional blocks with options for:
  - Grayscale or RGB input
  - Variable network depth (1-3 conv blocks)
  - Batch normalization
  - Different kernel counts

- **TextCNN**: Processes text through:
  - Word embeddings
  - Multiple parallel convolutional filters for n-gram capture
  - Global max pooling

- **MultimodalModel**: Combines image and text features through:
  - Feature concatenation
  - Fully connected fusion layers
  - Sigmoid activation for binary classification

## Requirements

- numpy
- tqdm
- Pillow
- datasets
- torch
- matplotlib

## Results

Experiment results are saved to:
- Training logs for Simple CNN: `training.log`
- Visualization plots: `experiment_plots/`(simple_cnn.py)  and `experiment_plots_2/`(advanced_cnn.py)