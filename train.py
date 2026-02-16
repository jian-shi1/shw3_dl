import torch
import numpy as np  # Добавлен импорт для перплексии
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
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend()

    # Перплексия = exp(loss) для каждого значения лосса
    train_perplexities = [np.exp(loss) for loss in train_losses]
    val_perplexities = [np.exp(loss) for loss in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')
    axs[1].set_xlabel('epoch')
    axs[1].legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0
    total_samples = 0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices.to(device)
        lengths = lengths.to(device)
        
        # Получаем логиты для ВСЕЙ последовательности (длина L)
        logits = model(indices, lengths)  # (B, L, V), где L = lengths.max()
        L = logits.size(1)
        
        # Цели: сдвигаем исходные индексы на 1 влево (предсказываем следующий токен)
        targets = indices[:, 1:L+1] if L < indices.size(1) else indices[:, 1:]  # (B, L)
        # Обрезаем targets до длины логитов (на случай, если L = max_length)
        targets = targets[:, :L]
        
        # Подготавливаем для CrossEntropyLoss
        logits_flat = logits.reshape(-1, logits.size(-1))  # (B*L, V)
        targets_flat = targets.reshape(-1)                  # (B*L,)
        
        loss = criterion(logits_flat, targets_flat)
        
        # Шаг оптимизации
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Накапливаем лосс * количество последовательностей в батче
        train_loss += loss.item() * indices.size(0)
        total_samples += indices.size(0)

    return train_loss / total_samples


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0
    total_samples = 0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices.to(device)
        lengths = lengths.to(device)
        
        logits = model(indices, lengths)  # (B, L, V)
        L = logits.size(1)
        
        targets = indices[:, 1:L+1] if L < indices.size(1) else indices[:, 1:]
        targets = targets[:, :L]
        
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        val_loss += loss.item() * indices.size(0)
        total_samples += indices.size(0)

    return val_loss / total_samples


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
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
        
        # Сохраняем чекпоинт после каждой эпохи
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'rnn_model_epoch_{epoch}.pt')
    
    # Сохраняем финальную модель
    torch.save(model.state_dict(), 'rnn_model_final.pt')
    print("Модель сохранена как 'rnn_model_final.pt'")