import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

def plot_losses(train_losses: List[float], val_losses: List[float]):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend()

    # Перплексия = exp(loss) — стандартная формула
    train_perplexities = [np.exp(l) for l in train_losses]
    val_perplexities = [np.exp(l) for l in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')
    axs[1].set_xlabel('epoch')
    axs[1].legend()
    plt.show()

def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    total_samples = 0

    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices.to(device)
        lengths = lengths.to(device)
        
        # КРИТИЧЕСКИ ВАЖНО: обрезаем lengths, чтобы L <= max_length - 1
        # Иначе indices[:, 1:L+1] выйдет за пределы при L = max_length
        max_possible = indices.size(1) - 1
        lengths = torch.clamp(lengths, max=max_possible)
        
        logits = model(indices, lengths)  # (B, L, V)
        L = logits.size(1)
        targets = indices[:, 1:L+1]        # (B, L) — гарантированно валидный срез
        
        # Подготовка к loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * indices.size(0)
        total_samples += indices.size(0)
    
    return total_loss / total_samples

@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices.to(device)
        lengths = lengths.to(device)
        
        # Та же обрезка lengths — критично для валидации
        max_possible = indices.size(1) - 1
        lengths = torch.clamp(lengths, max=max_possible)
        
        logits = model(indices, lengths)
        L = logits.size(1)
        targets = indices[:, 1:L+1]
        
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        total_loss += loss.item() * indices.size(0)
        total_samples += indices.size(0)
    
    return total_loss / total_samples

def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5):
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)
    device = next(model.parameters()).device
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())
        
        # Сохранение чекпоинта
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'rnn_model_epoch_{epoch}.pt')
    
    torch.save(model.state_dict(), 'rnn_model_final.pt')
    print("Модель сохранена как 'rnn_model_final.pt'")