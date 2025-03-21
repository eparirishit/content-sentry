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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

if torch.backends.mps.is_available():
    device = torch.device('mps')
    logging.info("Using Apple Silicon optimized Metal Performance Shaders (MPS) backend.")
else:
    device = torch.device('cpu')
    logging.info("MPS not available. Falling back to CPU.")

class HatefulMemesDataset:
    """
    Dataset class for the Hateful Memes dataset.
    
    Handles loading and preprocessing of image and text data.
    """
    def __init__(self, dataset, vocab, config, is_train=True):
        """
        Initialize the dataset.
        
        Args:
            dataset: Source dataset with image and text pairs
            vocab: Vocabulary mapping for text processing
            config: Configuration dictionary for image processing
            is_train: Whether this is training data (for augmentation)
        """
        self.dataset = dataset
        self.vocab = vocab
        self.config = config
        self.is_train = is_train
        self.max_length = 20
        self.img_size = config['img_size']
        
        # Image transforms
        self.grayscale = config['grayscale']
        self.channels = 1 if self.grayscale else 3

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def preprocess_image(self, img):
        """
        Preprocess image data for model input.
        
        Args:
            img: PIL Image object
            
        Returns:
            Processed image as numpy array
        """
        # Convert to grayscale/RGB and resize
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
            
        # Add simple data augmentation during training
        if self.is_train:
            # Random horizontal flip with 50% probability
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random brightness/contrast adjustments
            from PIL import ImageEnhance
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)
                
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
                
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img) / 255.0
        if self.grayscale:
            arr = arr[None, ...]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    def preprocess_text(self, text):
        """
        Preprocess text data for model input.
        
        Args:
            text: Raw text string
            
        Returns:
            Token indices as numpy array
        """
        tokens = text.lower().split()[:self.max_length]
        indices = [self.vocab.get(t, 1) for t in tokens]

        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        return np.array(indices)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (image, text, label)
        """
        item = self.dataset[idx]
        image = self.preprocess_image(item['image'])
        text = self.preprocess_text(item['text'])
        label = np.array([item['label']], dtype=np.float32)
        return image, text, label

class DataLoader:
    """
    Custom data loader for batched processing.
    
    Handles batching and optional shuffling of dataset examples.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        Initialize the data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of examples per batch
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        
    def __len__(self):
        """Return the number of batches in the dataset"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """
        Iterate through the dataset in batches.
        
        Yields:
            Tuple of (images, texts, labels) for each batch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = [int(idx) for idx in self.indices[i:i+self.batch_size]]
            images, texts, labels = [], [], []
            
            for idx in batch_indices:
                img, txt, lbl = self.dataset.__getitem__(idx)
                images.append(img)
                texts.append(txt)
                labels.append(lbl)
                
            yield (
                np.stack(images),
                np.stack(texts),
                np.stack(labels)
            )

def build_vocab(dataset, max_vocab=10000):
    """
    Build vocabulary mapping from dataset text.
    
    Args:
        dataset: Dataset with text examples
        max_vocab: Maximum vocabulary size
        
    Returns:
        Dictionary mapping words to indices
    """
    vocab = {'<pad>': 0, '<unk>': 1}
    word_counts = {}
    for example in dataset:
        tokens = example['text'].lower().split()
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    for word, _ in sorted_words[:max_vocab-2]:
        vocab[word] = len(vocab)
    return vocab

def safe_to_numpy(x):
    """
    Safely convert tensors to NumPy arrays regardless of device.
    
    Args:
        x: Tensor or array-like object
        
    Returns:
        NumPy array version of input
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def train(model_config=None):
    """
    Train a multimodal model on the Hateful Memes dataset.
    
    Args:
        model_config: Configuration dictionary for model architecture
        
    Returns:
        Dictionary of training metrics
    """
    # Load dataset
    ds = load_dataset("Multimodal-Fatima/Hatefulmemes_train")
    train_val = ds['train'].train_test_split(test_size=0.2)
    train_data = train_val['train']
    val_data = train_val['test']
    
    hateful = [i for i, item in enumerate(train_data) if item['label'] == 1]
    non_hateful = [i for i, item in enumerate(train_data) if item['label'] == 0]
    
    sample_size = min(3000, len(hateful), len(non_hateful))
    hateful_indices = np.random.choice(hateful, size=sample_size, replace=False)
    non_hateful_indices = np.random.choice(non_hateful, size=sample_size, replace=False)
    
    balanced_indices = np.concatenate([hateful_indices, non_hateful_indices])
    balanced_train_data = train_data.select(balanced_indices)
    
    sample_ratio = 0.5
    train_size = int(len(balanced_train_data) * sample_ratio)
    val_size = int(len(val_data) * sample_ratio)
    train_indices = np.random.choice(len(balanced_train_data), size=train_size, replace=False)
    val_indices = np.random.choice(len(val_data), size=val_size, replace=False)
    train_data = balanced_train_data.select(train_indices)
    val_data = val_data.select(val_indices)
    
    logging.info(f"Original dataset size: {len(train_data)} training, {len(val_data)} validation")
    logging.info(f"After balancing: {len(balanced_train_data)} training samples (equal class distribution)")
    logging.info(f"Final dataset size: {len(train_data)} training, {len(val_data)} validation")
    
    vocab = build_vocab(train_data, max_vocab=8000)
    
    if model_config is None:
        model_config = {
            'img_size': 32,
            'grayscale': True,
            'num_conv_blocks': 2,
            'num_kernels': 32
        }
    
    # Create datasets
    train_dataset = HatefulMemesDataset(train_data, vocab, model_config)
    val_dataset = HatefulMemesDataset(val_data, vocab, model_config, is_train=False)
    
    # Create models
    image_cnn = ImageCNN(model_config)
    text_cnn = TextCNN(len(vocab))
    model = MultimodalModel(image_cnn, text_cnn)

    model.image_model.device = device
    model.text_model.device = device
    
    logging.info("\nModel Architectures:")
    logging.info("Image CNN Architecture:")
    logging.info(image_cnn.layers)
    logging.info("\nText CNN Architecture:")
    logging.info(text_cnn.layers)
    logging.info("\nStarting training...")
        
    # Training parameters
    batch_size = 128
    epochs = 10
    lr = 0.01
    initial_lr = 0.01
    lr_scheduler = lambda epoch: initial_lr / (1 + 0.2 * epoch)
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


    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        correct = 0
        total = 0
        
        current_lr = lr_scheduler(epoch)
        logger.info(f"Current learning rate: {current_lr:.6f}")

        # Training
        model.train()
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        for batch_idx, (images, texts, labels) in enumerate(tqdm(train_loader, desc="Training")):
            # Convert inputs to tensors
            images = torch.tensor(images).to(device)
            texts = torch.tensor(texts).to(device)
            labels = torch.tensor(labels).to(device)
            
            # Forward pass
            preds = model.forward(safe_to_numpy(images), safe_to_numpy(texts))
            labels_np = safe_to_numpy(labels)
            preds_np = safe_to_numpy(preds)
            loss = loss_fn.eval(labels_np, preds_np)
            train_loss += loss
            
            # Backward pass
            grad = loss_fn.gradient(labels_np, preds_np)
            model.backward(grad)
            
            # Update weights with current learning rate
            model.fc.updateWeights(current_lr)
            model.image_model.update_weights(current_lr)
            model.text_model.update_weights(current_lr)
            
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
    """
    Plot the training and validation metrics for all experiments.
    
    Args:
        results: Dictionary mapping experiment names to metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    plt.savefig(f"{output_dir}/loss_comparison_{timestamp}.png", dpi=300)
    logging.info(f"Saved loss plot to {output_dir}/loss_comparison_{timestamp}.png")
    
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
        
        plt.savefig(f"{output_dir}/{name}accuracy{timestamp}.png", dpi=300)
        logging.info(f"Saved {name} accuracy plot to {output_dir}/{name}accuracy{timestamp}.png")

if __name__ == "__main__":
    experiments = [
        {"name": "Grayscale_SimpleCNN", "grayscale": True, "num_kernels": 8, "num_conv_blocks": 1, "use_batchnorm": False},
        {"name": "RGB_SimpleCNN", "grayscale": False, "num_kernels": 8, "num_conv_blocks": 1, "use_batchnorm": False},
        {"name": "Grayscale_MultiKernel", "grayscale": True, "num_kernels": 32, "num_conv_blocks": 1, "use_batchnorm": False},
        {"name": "RGB_DeepCNN", "grayscale": False, "num_kernels": 16, "num_conv_blocks": 3, "use_batchnorm": False},
    ]

    results = {}

    for config in experiments:
        logging.info(f"\n======= Starting experiment: {config['name']} =======")

        model_config = {
            'img_size': 64,
            'grayscale': config['grayscale'],
            'num_conv_blocks': config['num_conv_blocks'], 
            'num_kernels': config['num_kernels'],
            'use_batchnorm': config.get('use_batchnorm', False),
            'increase_channels': config.get('increase_channels', False)
        }
        
        metrics = train(model_config)
        results[config['name']] = metrics
        
        logging.info(f"\n======= Results for {config['name']} =======")
        logging.info(f"Training Accuracy: {metrics['train_accuracy']:.2%}")
        logging.info(f"Training Loss: {metrics['train_loss']:.4f}")
        logging.info(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
        logging.info(f"Validation Loss: {metrics['val_loss']:.4f}")
        logging.info(f"Epoch Duration: {metrics['epoch_duration']:.2f}s")
        logging.info("="*50)
        
        logging.info(f"Configuration details:")
        logging.info(f"  - Image size: {model_config['img_size']}px")
        logging.info(f"  - Image format: {'Grayscale' if model_config['grayscale'] else 'RGB'}")
        logging.info(f"  - CNN depth: {model_config['num_conv_blocks']} convolutional blocks")
        logging.info(f"  - Kernels per layer: {model_config['num_kernels']}")
    
    logging.info("\n======= EXPERIMENT SUMMARY =======")
    for name, metrics in results.items():
        logging.info(f"{name}: Val Acc={metrics['val_accuracy']:.2%}, Train Acc={metrics['train_accuracy']:.2%}, Time/epoch={sum(metrics['all_epoch_times'])/len(metrics['all_epoch_times']):.2f}s")
    
    logging.info("\n======= Generating visualization plots =======")
    plot_experiment_results(results)
    logging.info("Visualization complete. Plots saved to 'experiment_plots' directory.")