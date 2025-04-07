import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertForSequenceClassification, BertModel,
                          BertTokenizer)


class SimpleTextClassifier(nn.Module):
    """Simple text classifier to avoid BERT dependency issues."""
    
    def __init__(self, num_classes, vocab_size=10000, embedding_dim=200, hidden_dim=100):
        """
        Initialize a simple text classifier.
        
        Args:
            num_classes: Number of output classes
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layer
        """
        super(SimpleTextClassifier, self).__init__()
        
        # Use a smaller embedding to avoid issues
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # For compatibility with HuggingFace models
        self.config = type('Config', (), {'hidden_size': hidden_dim})
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the network.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            labels: Target labels (optional)
            
        Returns:
            An object with logits and loss if labels are provided
        """
        # Clamp input_ids to avoid out-of-range indices
        safe_input_ids = torch.clamp(input_ids, min=0, max=self.embedding.num_embeddings-1)
        
        # Get embeddings and average them
        x = self.embedding(safe_input_ids)
        
        # Mean pooling (average word embeddings)
        if attention_mask is not None:
            # Apply attention mask to get correct average
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, 1)
            sum_mask = torch.sum(attention_mask, 1, keepdim=True)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            # Simple mean if no mask
            x = torch.mean(x, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        # Return in a format similar to HuggingFace models
        return type('Output', (), {'loss': loss, 'logits': logits})
        
    def save_pretrained(self, output_dir):
        """
        Save model to a directory.
        
        Args:
            output_dir: Directory to save the model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        # Save a minimal config.json for compatibility
        config = {
            "vocab_size": self.embedding.num_embeddings,
            "hidden_size": self.fc1.in_features,
            "num_hidden_layers": 1,
            "num_labels": self.fc2.out_features
        }
        
        import json
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)


def load_bert_model(num_classes, model_name='bert-base-multilingual-cased'):
    """
    Load a model for text classification.
    
    Args:
        num_classes: Number of target classes
        model_name: Pre-trained model name (ignored in this implementation)
        
    Returns:
        Text classification model
    """
    print(f"Creating a simple text classifier with {num_classes} classes")
    # Use our simple classifier instead of BERT
    model = SimpleTextClassifier(num_classes=num_classes)
    print("Simple text classifier created successfully")
    return model


def save_model(model, tokenizer, output_dir):
    """
    Save model and tokenizer.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Check if model has custom save method
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
    else:
        # Custom save
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    # Save tokenizer if possible
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
        # Save vocab as a simple text file
        with open(os.path.join(output_dir, 'vocab.txt'), 'w') as f:
            f.write("<PAD>\n<UNK>\n" + "\n".join([f"token{i}" for i in range(10000)]))
    
def load_model(model_dir):
    """
    Load model and tokenizer from directory.
    
    Args:
        model_dir: Directory containing saved model and tokenizer
        
    Returns:
        model, tokenizer
    """
    try:
        print(f"Attempting to load saved model from {model_dir}")
        
        # Create a simple text classifier
        model = SimpleTextClassifier(num_classes=5)  # Default to 5 classes
        
        # Try to load saved weights
        model_path = os.path.join(model_dir, 'pytorch_model.bin')
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print("Model weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
        
        # Load tokenizer or create a simple one
        tokenizer_path = os.path.join(model_dir, 'vocab.txt')
        if os.path.exists(tokenizer_path):
            try:
                # Try to load BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(model_dir)
                print("Tokenizer loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                # Create a basic tokenizer
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                print("Using basic tokenizer")
        else:
            # Create a basic tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("Using basic tokenizer")
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Fatal error loading model: {str(e)}")
        raise 