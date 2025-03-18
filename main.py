import numpy as np
import time
import logging
import numpy as np

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from models.image_cnn import ImageCNN
from models.text_cnn import TextCNN
from models.multimodal import MultimodalModel
from framework import CrossEntropy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


class HatefulMemesDataset:
    def __init__(self, dataset, vocab, config, is_train=True):
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
        return len(self.dataset)

    def preprocess_image(self, img):
        # Convert to grayscale/RGB and resize
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img) / 255.0
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

def train():
    # Load dataset
    ds = load_dataset("Multimodal-Fatima/Hatefulmemes_train")
    train_val = ds['train'].train_test_split(test_size=0.2)
    train_data = train_val['train']
    val_data = train_val['test']
    
    # Build vocabulary
    vocab = build_vocab(train_data)
    
    # Model configurations
    config = {
        'img_size': 64,
        'grayscale': False,
        'num_conv_blocks': 2,
        'num_kernels': 32
    }
    
    # Create datasets
    train_dataset = HatefulMemesDataset(train_data, vocab, config)
    val_dataset = HatefulMemesDataset(val_data, vocab, config, is_train=False)
    
    # Create models
    image_cnn = ImageCNN(config)
    text_cnn = TextCNN(len(vocab))
    model = MultimodalModel(image_cnn, text_cnn)
    
    logging.info("\nModel Architectures:")
    logging.info("Image CNN Architecture:")
    logging.info(image_cnn.layers)
    logging.info("\nText CNN Architecture:")
    logging.info(text_cnn.layers)
    logging.info("\nStarting training...")
        
    # Training parameters
    batch_size = 64
    epochs = 10
    lr = 0.001
    loss_fn = CrossEntropy()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger = logging.getLogger()
    logger.info("Starting training with:")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Image size: {config['img_size']}")
    logger.info(f"Conv blocks: {config['num_conv_blocks']}")

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0
        correct = 0
        total = 0

        # Training
        model.train()
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        for batch_idx, (images, texts, labels) in enumerate(tqdm(train_loader, desc="Training")):
            # Forward pass
            preds = model.forward(images, texts)
            loss = loss_fn.eval(labels, preds)
            train_loss += loss
            
            # Backward pass
            grad = loss_fn.gradient(labels, preds)
            model.backward(grad)

            # Update weights
            model.fc.updateWeights(lr)
            model.image_model.update_weights(lr)
            model.text_model.update_weights(lr)

            # Calculate metrics
            correct += ((preds > 0.5) == labels).sum()
            total += labels.shape[0]
            
            # Batch logging
            if (batch_idx + 1) % 50 == 0:
                batch_acc = correct / total
                logger.debug(f"Batch {batch_idx+1} | Loss: {loss:.4f} | Acc: {batch_acc:.2f}")
            
        # Validation phase
        val_start = time.time()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        model.eval()
        for images, texts, labels in tqdm(val_loader, desc="Validation"):
            preds = model.forward(images, texts)
            val_loss += loss_fn.eval(labels, preds)
            val_correct += ((preds > 0.5) == labels).sum()
            val_total += labels.shape[0]
        
        # Epoch statistics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        epoch_duration = time.time() - epoch_start
        
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Duration: {epoch_duration:.2f}s")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2%}")
        logger.info(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2%}")
        logger.info("-" * 50)

if __name__ == "__main__":
    train()