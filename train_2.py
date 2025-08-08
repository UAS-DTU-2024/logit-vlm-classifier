import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim
from torch.optim.lr_scheduler import StepLR
import gc
from new_dataset_soft import MLLMDataset, custom_collate_fxn
from models import reduced_classifier_moreblocks
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import wandb
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


# tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")


want_to_resume = False


CHECKPOINT_PATH = "/home/uas-dtu/llava-interp-main/second_experiment/checkpointnew.pth"
img_dir = "/home/uas-dtu/Desktop/segmented_images/train_data"
csv_path = "/home/uas-dtu/llava-interp-main/sorted_csv.csv"
embeddings_file = "/media/uas-dtu/e117d1c8-5ec6-446b-8141-37a5493e1fb2/FULL_precomputed_embeddings.h5"

device = "cuda:0"
num_epochs = 30
torch.cuda.empty_cache()
dataset = MLLMDataset(img_dir, csv_path, embeddings_file)

labels = dataset.current_df["trauma"].tolist()
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_indices, val_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.1,
    stratify=labels,
    random_state=42,
    shuffle=True
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=custom_collate_fxn,
    num_workers=16
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fxn,
    num_workers=16
)
print("Length of train dataset",len(train_dataset))
print("Length of val dataset",len(val_dataset))

train_labels = [labels[i] for i in train_indices]
val_labels = [labels[i] for i in val_indices]

print("Train label distribution:", Counter(train_labels))
print("Val label distribution:", Counter(val_labels))

model = reduced_classifier_moreblocks(dim=4096, num_heads=8, num_blocks=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

best_val_loss = float('inf') 

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
start_epoch = 0

wandb.init(
    project="Chirag_sir_pls_review",
    config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "architecture": "reduced_classifier_moreblocks",
        "dataset": "Dataset-Trauma",
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau"
    }
)

if want_to_resume and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming with best validation loss: {best_val_loss:.4f}")
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found or specified. Starting training from scratch.")

def print_trainable_parameters(model):
    total_params = 0
    print("\nTrainable parameters:\n" + "-" * 40)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60} | {tuple(param.shape)} | {param.numel():,}")
            total_params += param.numel()
    print("-" * 40)
    print(f"Total trainable parameters: {total_params:,}\n")

print_trainable_parameters(model)


for epoch in range(start_epoch, num_epochs):

    dataset.resample()

    model.train()
    running_loss = 0.0
    correct_total = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

    for batch_idx, (images, labels, hidden_states, mask, image_names) in enumerate(progress_bar):
        try:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            batch_cls_embeds = hidden_states
            batch_cls_embeds = batch_cls_embeds.to(device)
            labels=labels.to(device)

            B, S = mask.shape

            mask = mask.to(device)
            cls_token_mask = torch.ones(B, 1, device=device)
            mask = torch.cat([cls_token_mask, mask], dim=1)
            
            optimizer.zero_grad()

            outputs, attention_weights = model(batch_cls_embeds, attention_mask=mask, output_attentions=True)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1) 
                        
            correct = (preds == labels).sum().item()
            correct_total += correct

            batch_loss = loss.item()
            batch_acc = correct / labels.size(0)

            total += labels.size(0)

            progress_bar.set_postfix({
                'Batch Loss': f'{batch_loss:.4f}',
                'Batch Acc': f'{batch_acc:.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Avg Acc': f'{correct_total/total:.4f}'
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA OOM at training batch {batch_idx}. Clearing cache and continuing...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    epoch_loss = running_loss / (batch_idx + 1)
    train_acc = correct_total / max(total, 1)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (images, labels, hidden_states, mask, image_names) in enumerate(val_loader):
            try:
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                batch_cls_embeds = hidden_states
                batch_cls_embeds = batch_cls_embeds.to(device)
                labels=labels.to(device)

                B, S = mask.shape
                mask = mask.to(device)
                cls_token_mask = torch.ones(B, 1, device=device)
                mask = torch.cat([cls_token_mask, mask], dim=1)

                outputs, attention_weights = model(batch_cls_embeds, attention_mask=mask, output_attentions=True)

                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1) 
                            
                correct = (preds == labels).sum().item()
                val_correct += correct
                
                accuracy = correct/labels.size(0)
                
                val_total += labels.size(0)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM in validation at batch {batch_idx}. Skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    val_loss = val_running_loss / (batch_idx + 1)
    val_acc = val_correct / max(val_total, 1)
    scheduler.step(val_loss)

    print(f"\nEpoch {epoch+1} Summary - Train: Loss={epoch_loss:.4f}, Acc={train_acc:.4f} ({correct_total}/{total})")
    print(f"                   -   Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} ({val_correct}/{val_total})")

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": epoch_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        best_model_path = 'best_model.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"  NEW BEST MODEL SAVED! Best Val loss: {best_val_loss:.4f}")
    else:
        print(f"  Current Best Val loss: {best_val_loss:.4f}")
    
    checkpoint_path = f"latest_epoch_weights.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'best_val_loss': best_val_loss
    }, checkpoint_path)
    
    torch.cuda.empty_cache()
    gc.collect()

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"Training History": wandb.Image(plt)})
    plt.savefig('review.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining Summary:")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(val_accuracies):.4f} (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")

print("\n" + "=" * 60)
print(" TRAINING COMPLETED SUCCESSFULLY! ")
print(f"Final Best Validation Loss: {best_val_loss:.4f}")
print("=" * 60)

print("+" * 25)
print(train_losses)
print("+" * 25)
print(val_losses)

plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

wandb.finish()