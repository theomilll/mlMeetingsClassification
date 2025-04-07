import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SummaryDataset(Dataset):
    """Dataset for meeting summary categorization."""
    
    def __init__(self, encodings, labels):
        """
        Initialize dataset.
        
        Args:
            encodings: Pre-encoded inputs from tokenizer
            labels: List of corresponding labels
        """
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load and preprocess data for training.
    
    Args:
        data_path: Path to CSV file with 'text' and 'category' columns
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_texts, val_texts, train_labels, val_labels, label_encoder
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    if 'text' not in df.columns or 'category' not in df.columns:
        raise ValueError("Data file must contain 'text' and 'category' columns")
    
    # Create label encoder
    categories = df['category'].unique().tolist()
    label_encoder = {category: idx for idx, category in enumerate(categories)}
    
    # Encode labels
    df['label'] = df['category'].map(label_encoder)
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['category']
    )
    
    print(f"Training data: {len(train_df)} examples")
    print(f"Validation data: {len(val_df)} examples")
    print(f"Categories: {categories}")
    
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    
    return train_texts, val_texts, train_labels, val_labels, label_encoder

def prepare_datasets(train_texts, val_texts, train_labels, val_labels, model_name='bert-base-multilingual-cased'):
    """
    Prepare PyTorch datasets for training and validation.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        train_labels: Training labels
        val_labels: Validation labels
        model_name: Pre-trained model name
        
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    # Load tokenizer
    try:
        print(f"Loading tokenizer for {model_name}...")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        # Fall back to basic BERT tokenizer if there's an issue
        print("Falling back to basic tokenizer implementation")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                                 do_lower_case=False,
                                                 local_files_only=False)
    
    # Tokenize texts
    print("Tokenizing training texts...")
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=128,
        return_tensors='pt'
    )
    
    print("Tokenizing validation texts...")
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=128,
        return_tensors='pt'
    )
    
    # Create datasets
    train_dataset = SummaryDataset(train_encodings, train_labels)
    val_dataset = SummaryDataset(val_encodings, val_labels)
    
    return train_dataset, val_dataset, tokenizer

def save_label_encoder(label_encoder, output_dir):
    """
    Save the label encoder to a file.
    
    Args:
        label_encoder: Dictionary mapping categories to indices
        output_dir: Directory to save the encoder
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create reverse mapping
    idx_to_category = {idx: category for category, idx in label_encoder.items()}
    
    # Save to CSV
    encoder_df = pd.DataFrame(
        list(idx_to_category.items()), 
        columns=['idx', 'category']
    )
    encoder_df.to_csv(os.path.join(output_dir, 'label_encoder.csv'), index=False)
    
    return os.path.join(output_dir, 'label_encoder.csv')

def verify_data_format(data_path):
    """
    Verify that the data file has the correct format.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        True if the data file has the correct format, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"Error: File {data_path} does not exist")
            return False
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Check if required columns exist
        if 'text' not in df.columns or 'category' not in df.columns:
            print("Error: Data file must contain 'text' and 'category' columns")
            return False
        
        # Check if there are at least 2 categories
        categories = df['category'].unique()
        if len(categories) < 2:
            print(f"Error: Data file must contain at least 2 categories, found {len(categories)}")
            return False
        
        # Check if there are at least 2 examples per category
        for category in categories:
            count = len(df[df['category'] == category])
            if count < 2:
                print(f"Error: Category '{category}' has only {count} example(s), need at least 2")
                return False
        
        print("Data format verification passed!")
        print(f"Total examples: {len(df)}")
        print(f"Categories: {categories.tolist()}")
        print("Examples per category:")
        for category in categories:
            count = len(df[df['category'] == category])
            print(f"  {category}: {count}")
        
        return True
    
    except Exception as e:
        print(f"Error verifying data format: {e}")
        return False 