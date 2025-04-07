import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data_preprocessing import (load_and_preprocess_data, prepare_datasets,
                                save_label_encoder, verify_data_format)
from evaluate import compute_metrics
from model import load_bert_model, save_model


def train_model(model, train_dataloader, val_dataloader, device, 
                num_epochs=3, lr=2e-5, warmup_steps=0, output_dir='./models'):
    """
    Train the model.
    
    Args:
        model: BERT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to use (cpu/cuda)
        num_epochs: Number of training epochs
        lr: Learning rate
        warmup_steps: Learning rate warmup steps
        output_dir: Directory to save model checkpoints
        
    Returns:
        Trained model
    """
    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Total steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Overall progress bar for epochs
    print("\n=== Starting Training ===")
    print(f"Total training steps: {total_steps}")
    overall_progress = tqdm(total=total_steps, desc="Overall Progress", position=0)
    steps_completed = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} ({((epoch+1)/num_epochs)*100:.1f}% complete)")
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update loss
            train_loss += loss.item()
            
            # Update progress bars
            steps_completed += 1
            overall_progress.update(1)
            epoch_progress = ((steps_completed % len(train_dataloader)) / len(train_dataloader)) * 100
            total_progress = (steps_completed / total_steps) * 100
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'epoch progress': f"{epoch_progress:.1f}%",
                'total progress': f"{total_progress:.1f}%"
            })
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        print("Running validation...")
        val_progress = tqdm(val_dataloader, desc="Validation", position=1, leave=False)
        
        with torch.no_grad():
            for batch in val_progress:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # Update loss
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                val_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate metrics
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        val_accuracy = metrics['accuracy']
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {metrics['f1']:.4f}")
        
        # Save model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-epoch-{epoch+1}')
            
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            # Save model
            model.save_pretrained(checkpoint_dir)
            
            print(f"Model checkpoint saved: {checkpoint_dir}")
    
    overall_progress.close()
    print("\n=== Training Complete ===")
    
    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a BERT model for meeting summary categorization")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save the model')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', help='Pre-trained model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Verify data format
    print("\nVerifying data format...")
    if not verify_data_format(args.data_path):
        print("Data verification failed. Please check your data file and try again.")
        return
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(
        args.data_path, test_size=args.test_size, random_state=args.seed
    )
    
    # Save label encoder
    encoder_path = save_label_encoder(label_encoder, args.model_dir)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset, tokenizer = prepare_datasets(
        train_texts, val_texts, train_labels, val_labels, model_name=args.model_name
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    # Load model
    print("\nLoading model...")
    num_classes = len(label_encoder)
    model = load_bert_model(num_classes, model_name=args.model_name)
    model = model.to(device)
    
    # Train model
    print("\nTraining model...")
    model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs=args.epochs,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        output_dir=args.model_dir
    )
    
    # Save final model
    final_model_dir = os.path.join(args.model_dir, 'final')
    
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    save_model(model, tokenizer, final_model_dir)
    print(f"Final model saved to: {final_model_dir}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 