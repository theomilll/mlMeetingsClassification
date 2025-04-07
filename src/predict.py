import logging
import os

import pandas as pd
import torch

from evaluate import load_label_encoder
from model import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryClassifier:
    """Classifier for meeting summaries."""
    
    def __init__(self, model_dir):
        """
        Initialize classifier.
        
        Args:
            model_dir: Directory containing the model and label encoder
        """
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.model, self.tokenizer = load_model(model_dir)
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Find and load label encoder - look in several possible locations
        potential_paths = [
            os.path.join(model_dir, 'label_encoder.csv'),                 # In the model directory itself
            os.path.join(os.path.dirname(model_dir), 'label_encoder.csv'),  # In the parent directory
            os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'label_encoder.csv')  # In grandparent directory
        ]
        
        encoder_path = None
        for path in potential_paths:
            if os.path.exists(path):
                encoder_path = path
                break
                
        if encoder_path:
            try:
                self.label_encoder = load_label_encoder(encoder_path)
                # Create reverse mapping
                self.idx_to_category = {idx: category for category, idx in self.label_encoder.items()}
                logger.info(f"Label encoder loaded from {encoder_path}")
                logger.info(f"Available categories: {list(self.label_encoder.keys())}")
            except Exception as e:
                logger.error(f"Failed to load label encoder: {e}")
                raise RuntimeError(f"Failed to load label encoder: {e}")
        else:
            logger.error(f"Label encoder not found in any expected location")
            raise FileNotFoundError(f"Label encoder not found. Searched in: {', '.join(potential_paths)}")
    
    def classify(self, text, return_confidence=False):
        """
        Classify a text summary.
        
        Args:
            text: Text to classify
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted category and optionally confidence scores
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Prepare input
        try:
            encoding = self.tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Set model to eval mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predicted class
                _, predicted_idx = torch.max(logits, dim=1)
                predicted_idx = predicted_idx.item()
                
                # Check if the index is in our mapping
                if predicted_idx not in self.idx_to_category:
                    logger.error(f"Predicted index {predicted_idx} not found in label mapping")
                    raise RuntimeError(f"Invalid prediction index: {predicted_idx}")
                
                predicted_category = self.idx_to_category[predicted_idx]
                
                if return_confidence:
                    # Apply softmax to get probabilities
                    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                    
                    # Create dictionary of category -> confidence
                    confidence = {}
                    for idx, prob in enumerate(probabilities):
                        if idx in self.idx_to_category:
                            category = self.idx_to_category[idx]
                            confidence[category] = prob.item()
                    
                    return predicted_category, confidence
                
                return predicted_category
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

def classify_summary(text, model_dir='./models/final'):
    """
    Classify a summary.
    
    Args:
        text: Text to classify
        model_dir: Directory containing the model
        
    Returns:
        Predicted category
    """
    try:
        classifier = SummaryClassifier(model_dir)
        return classifier.classify(text)
    except Exception as e:
        logger.error(f"Error in classify_summary: {e}")
        raise

def classify_batch(texts, model_dir='./models/final'):
    """
    Classify a batch of summaries.
    
    Args:
        texts: List of texts to classify
        model_dir: Directory containing the model
        
    Returns:
        List of predicted categories
    """
    try:
        # Input validation
        if not isinstance(texts, list):
            raise ValueError("Texts must be a list")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")
        
        # Initialize classifier once to avoid loading the model multiple times
        classifier = SummaryClassifier(model_dir)
        
        # Classify each text
        return [classifier.classify(text) for text in texts]
    except Exception as e:
        logger.error(f"Error in classify_batch: {e}")
        raise

def classify_with_confidence(text, model_dir='./models/final'):
    """
    Classify a summary and return confidence scores.
    
    Args:
        text: Text to classify
        model_dir: Directory containing the model
        
    Returns:
        Tuple of (predicted category, confidence scores)
    """
    try:
        classifier = SummaryClassifier(model_dir)
        return classifier.classify(text, return_confidence=True)
    except Exception as e:
        logger.error(f"Error in classify_with_confidence: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify a meeting summary")
    parser.add_argument('text', type=str, help='Text to classify')
    parser.add_argument('--model_dir', type=str, default='./models/final',
                       help='Directory containing the model')
    parser.add_argument('--confidence', action='store_true',
                       help='Return confidence scores')
    
    args = parser.parse_args()
    
    try:
        if args.confidence:
            category, confidence = classify_with_confidence(args.text, args.model_dir)
            print(f"Category: {category}")
            print("Confidence scores:")
            for cat, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {score:.4f}")
        else:
            category = classify_summary(args.text, args.model_dir)
            print(f"Category: {category}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1) 