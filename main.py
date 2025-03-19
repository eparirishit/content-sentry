import numpy as np
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from models.image_cnn import ImageCNN
from models.text_cnn import TextCNN
from models.multimodal import MultimodalModel
from framework import CrossEntropy
import random
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Configure device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    logging.info("Using Apple Silicon optimized Metal Performance Shaders (MPS) backend.")
else:
    device = torch.device('cpu')
    logging.info("MPS not available. Falling back to CPU.")

class HatefulMemesDataset:
    def __init__(self, dataset, vocab, config, is_train=True):
        self.dataset = dataset
        self.vocab = vocab
        self.config = config
        self.is_train = is_train
        self.max_length = 30  # Increased from 20
        self.img_size = config['img_size']
        
        # Image transforms
        self.grayscale = config['grayscale']
        self.channels = 1 if self.grayscale else 3
        
        # Data augmentation for training
        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        else:
            self.transforms = None

    def __len__(self):
        return len(self.dataset)

    def preprocess_image(self, img):
        # Apply data augmentation if in training mode
        if self.is_train and self.transforms:
            img = self.transforms(img)
            
        # Convert to grayscale/RGB and resize
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img) / 255.0
        
        # Standardize images (normalize to mean=0, std=1)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        
        if self.grayscale:
            arr = arr[None, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    def preprocess_text(self, text):
        tokens = text.lower().split()[:self.max_length]
        indices = [self.vocab.get(t, 1) for t in tokens]

        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        return np.array(indices)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.preprocess_image(item['image'])
        text = self.preprocess_text(item['text'])
        label = np.array([item['label']], dtype=np.float32)
        return image, text, label

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        
    def __len__(self):
        """Return the number of batches in the dataset"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = [int(idx) for idx in self.indices[i:i+self.batch_size]]
            images, texts, labels = [], [], []
            
            for idx in batch_indices:
                img, txt, lbl = self.dataset[idx]
                images.append(img)
                texts.append(txt)
                labels.append(lbl)
                
            yield (
                np.stack(images),
                np.stack(texts),
                np.stack(labels)
            )

def build_vocab(dataset, max_vocab=10000):
    vocab = {'<pad>': 0, '<unk>': 1}
    word_counts = {}
    for example in dataset:
        tokens = example['text'].lower().split()
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    for word, _ in sorted_words[:max_vocab-2]:
        vocab[word] = len(vocab)
    return vocab

def safe_to_numpy(x):
    """Safely convert tensors to NumPy arrays regardless of device"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def train(model_config=None):
    # Load dataset
    ds = load_dataset("Multimodal-Fatima/Hatefulmemes_train")
    train_val = ds['train'].train_test_split(test_size=0.2)
    train_data = train_val['train']
    val_data = train_val['test']
    
    # First, balance the dataset to have 3000 entries for each label
    hateful = [i for i, item in enumerate(train_data) if item['label'] == 1]
    non_hateful = [i for i, item in enumerate(train_data) if item['label'] == 0]
    
    # Sample 3000 from each class (or all if less than 3000)
    sample_size = min(3000, len(hateful), len(non_hateful))
    hateful_indices = np.random.choice(hateful, size=sample_size, replace=False)
    non_hateful_indices = np.random.choice(non_hateful, size=sample_size, replace=False)
    
    # Combine the balanced indices and create a new balanced dataset
    balanced_indices = np.concatenate([hateful_indices, non_hateful_indices])
    balanced_train_data = train_data.select(balanced_indices)
    
    # For faster training - use 50% of the balanced data
    sample_ratio = 0.5  # Use 50% of data (increased from 25%)
    train_size = int(len(balanced_train_data) * sample_ratio)
    val_size = int(len(val_data) * sample_ratio)
    train_indices = np.random.choice(len(balanced_train_data), size=train_size, replace=False)
    val_indices = np.random.choice(len(val_data), size=val_size, replace=False)
    train_data = balanced_train_data.select(train_indices)
    val_data = val_data.select(val_indices)
    
    # Log dataset statistics
    logging.info(f"Original dataset size: {len(train_data)} training, {len(val_data)} validation")
    logging.info(f"After balancing: {len(balanced_train_data)} training samples (equal class distribution)")
    logging.info(f"Final dataset size: {len(train_data)} training, {len(val_data)} validation")
    
    # Build vocabulary with smaller size
    vocab = build_vocab(train_data, max_vocab=8000)  # Reduced from 10000
    
    # Use provided model config or default
    if model_config is None:
        model_config = {
            'img_size': 128,  # Increased from 32
            'grayscale': True,
            'num_conv_blocks': 3,  # Increased from 2
            'num_kernels': 64,  # Increased from 32
            'dropout_rate': 0.5,
            'use_batch_norm': True
        }
    
    # Create datasets
    train_dataset = HatefulMemesDataset(train_data, vocab, model_config)
    val_dataset = HatefulMemesDataset(val_data, vocab, model_config, is_train=False)
    
    # Create models
    image_cnn = ImageCNN(model_config)
    text_cnn = TextCNN(len(vocab))
    model = MultimodalModel(image_cnn, text_cnn)

    # Ensure all model parameters and tensors are moved to the correct device
    model.image_model.device = device
    model.text_model.device = device
    
    logging.info("\nModel Architectures:")
    logging.info("Image CNN Architecture:")
    logging.info(image_cnn.layers)
    logging.info("\nText CNN Architecture:")
    logging.info(text_cnn.layers)
    logging.info("\nStarting training...")
        
    # Training parameters
    batch_size = 64  # Reduced for better stability
    epochs = 10  # Increased from 5
    lr = 0.001  # Reduced from 0.01
    loss_fn = CrossEntropy()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger = logging.getLogger()
    logger.info("Starting training with:")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Image size: {model_config['img_size']}")
    logger.info(f"Conv blocks: {model_config['num_conv_blocks']}")
    logger.info(f"Grayscale: {model_config['grayscale']}")
    logger.info(f"Num kernels: {model_config['num_kernels']}")
    logger.info(f"Using {sample_ratio*100:.0f}% of dataset: {train_size} training, {val_size} validation samples")

    # Track metrics per epoch for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    
    # Initialize best validation accuracy for early stopping
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    # Learning rate scheduler: reduce LR by 0.5 every 3 epochs
    lr_schedule = {3: 0.5, 6: 0.5, 9: 0.5}
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        correct = 0
        total = 0
        
        # Apply learning rate schedule
        if epoch in lr_schedule:
            lr *= lr_schedule[epoch]
            logger.info(f"Reducing learning rate to {lr}")

        # Training
        model.train()
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        for batch_idx, (images, texts, labels) in enumerate(tqdm(train_loader, desc="Training")):
            # Convert inputs to tensors on the right device
            images = torch.tensor(images).to(device)
            texts = torch.tensor(texts).to(device)
            labels = torch.tensor(labels).to(device)
            
            # Forward pass - ensure we're passing CPU numpy arrays where needed
            preds = model.forward(safe_to_numpy(images), safe_to_numpy(texts))
            labels_np = safe_to_numpy(labels)
            preds_np = safe_to_numpy(preds)
            loss = loss_fn.eval(labels_np, preds_np)
            train_loss += loss
            
            # Backward pass with numpy arrays
            grad = loss_fn.gradient(labels_np, preds_np)
            model.backward(grad)
            
            # Update weights with Adam-like momentum
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            
            # Initialize momentum and velocity if not present
            if not hasattr(model, 'momentum'):
                model.momentum = {}
                model.velocity = {}
                
            # Update with Adam-like momentum (simplified version)
            model.fc.updateWeightsWithMomentum(lr, beta1, beta2, epsilon, model.momentum, model.velocity)
            model.image_model.update_weights_with_momentum(lr, beta1, beta2, epsilon, model.momentum, model.velocity)
            model.text_model.update_weights_with_momentum(lr, beta1, beta2, epsilon, model.momentum, model.velocity)
            
            # Calculate metrics - properly move to CPU before NumPy conversion
            correct += ((preds_np > 0.5) == labels_np).sum()
            total += labels.shape[0]

        # Validation phase
        val_start = time.time()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        model.eval()
        for images, texts, labels in tqdm(val_loader, desc="Validation"):
            images = torch.tensor(images).to(device)
            texts = torch.tensor(texts).to(device)
            labels = torch.tensor(labels).to(device)
            
            preds = model.forward(safe_to_numpy(images), safe_to_numpy(texts))
            labels_np = safe_to_numpy(labels)
            preds_np = safe_to_numpy(preds)
            val_loss += loss_fn.eval(labels_np, preds_np)
            val_correct += ((preds_np > 0.5) == labels_np).sum()
            val_total += labels.shape[0]
        
        # Epoch statistics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        epoch_duration = time.time() - epoch_start
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        epoch_times.append(epoch_duration)
        
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Duration: {epoch_duration:.2f}s")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2%}")
        logger.info(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2%}")
        logger.info("-" * 50)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model weights here if needed
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Return final validation metrics
    final_metrics = {
        'val_loss': avg_val_loss,
        'val_accuracy': val_acc,
        'train_loss': avg_train_loss, 
        'train_accuracy': train_acc,
        'epoch_duration': epoch_duration,
        'all_train_losses': train_losses,
        'all_val_losses': val_losses,
        'all_train_accs': train_accs,
        'all_val_accs': val_accs,
        'all_epoch_times': epoch_times
    }
    
    return final_metrics

def plot_experiment_results(results, output_dir="experiment_plots"):
    """Plot the training and validation accuracies for all experiments"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training and validation accuracies for each experiment
    plt.figure(figsize=(12, 8))
    
    for name, metrics in results.items():
        epochs = range(1, len(metrics['all_train_accs']) + 1)
        plt.plot(epochs, metrics['all_train_accs'], '-o', label=f"{name} (Train)")
        plt.plot(epochs, metrics['all_val_accs'], '-s', label=f"{name} (Val)")
    
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/accuracy_comparison_{timestamp}.png", dpi=300)
    logging.info(f"Saved accuracy plot to {output_dir}/accuracy_comparison_{timestamp}.png")
    
    # Plot training and validation losses for each experiment
    plt.figure(figsize=(12, 8))
    
    for name, metrics in results.items():
        epochs = range(1, len(metrics['all_train_losses']) + 1)
        plt.plot(epochs, metrics['all_train_losses'], '-o', label=f"{name} (Train)")
        plt.plot(epochs, metrics['all_val_losses'], '-s', label=f"{name} (Val)")
    
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/loss_comparison_{timestamp}.png", dpi=300)
    logging.info(f"Saved loss plot to {output_dir}/loss_comparison_{timestamp}.png")
    
    # Also plot individual experiment metrics for clearer visualization
    for name, metrics in results.items():
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(metrics['all_train_accs']) + 1)
        
        plt.plot(epochs, metrics['all_train_accs'], '-o', label='Training Accuracy')
        plt.plot(epochs, metrics['all_val_accs'], '-s', label='Validation Accuracy')
        
        plt.title(f'Model Accuracy: {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
        
        # Save the figure
        plt.savefig(f"{output_dir}/{name}_accuracy_{timestamp}.png", dpi=300)
        logging.info(f"Saved {name} accuracy plot to {output_dir}/{name}_accuracy_{timestamp}.png")

if __name__ == "__main__":
    # Define experiments to run with improved configurations
    experiments = [
        {"name": "Grayscale_DeepCNN", "grayscale": True, "num_kernels": 32, "num_conv_blocks": 3, 
         "dropout_rate": 0.5, "use_batch_norm": True},
        {"name": "RGB_DeepCNN", "grayscale": False, "num_kernels": 32, "num_conv_blocks": 3,
         "dropout_rate": 0.5, "use_batch_norm": True},
        {"name": "Grayscale_VeryDeepCNN", "grayscale": True, "num_kernels": 64, "num_conv_blocks": 4,
         "dropout_rate": 0.5, "use_batch_norm": True},
        {"name": "RGB_VeryDeepCNN", "grayscale": False, "num_kernels": 64, "num_conv_blocks": 4,
         "dropout_rate": 0.5, "use_batch_norm": True}
    ]

    # Results dictionary
    results = {}

    for config in experiments:
        logging.info(f"\n======= Starting experiment: {config['name']} =======")
        # Set up the configuration
        model_config = {
            'img_size': 128,  # Increased from 64
            'grayscale': config['grayscale'],
            'num_conv_blocks': config['num_conv_blocks'], 
            'num_kernels': config['num_kernels'],
            'dropout_rate': config['dropout_rate'],
            'use_batch_norm': config['use_batch_norm']
        }
        
        # Train and evaluate with this configuration
        metrics = train(model_config)
        
        # Store results
        results[config['name']] = metrics
        
        logging.info(f"\n======= Results for {config['name']} =======")
        logging.info(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
        logging.info(f"Validation Loss: {metrics['val_loss']:.4f}")
        logging.info(f"Training Accuracy: {metrics['train_accuracy']:.2%}")
        logging.info(f"Training Loss: {metrics['train_loss']:.4f}")
        logging.info(f"Epoch Duration: {metrics['epoch_duration']:.2f}s")
        logging.info("="*50)
        
        # Log more detailed configuration info
        logging.info(f"Configuration details:")
        logging.info(f"  - Image size: {model_config['img_size']}px")
        logging.info(f"  - Image format: {'Grayscale' if model_config['grayscale'] else 'RGB'}")
        logging.info(f"  - CNN depth: {model_config['num_conv_blocks']} convolutional blocks")
        logging.info(f"  - Kernels per layer: {model_config['num_kernels']}")
    
    # Print summary of all experiments
    logging.info("\n======= EXPERIMENT SUMMARY =======")
    for name, metrics in results.items():
        logging.info(f"{name}: Val Acc={metrics['val_accuracy']:.2%}, Train Acc={metrics['train_accuracy']:.2%}, Time/epoch={sum(metrics['all_epoch_times'])/len(metrics['all_epoch_times']):.2f}s")
    
    # Generate plots
    logging.info("\n======= Generating visualization plots =======")
    plot_experiment_results(results)
    logging.info("Visualization complete. Plots saved to 'experiment_plots' directory.")