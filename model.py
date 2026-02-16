import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        super(LanguageModel, self).__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        # Создаём слои
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_size,
            padding_idx=dataset.pad_id
        )
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # КРИТИЧЕСКИ ВАЖНО: обрезаем индексы до валидного диапазона
        indices = torch.clamp(indices, min=0, max=self.vocab_size - 1)
        
        max_len = int(lengths.max().item())
        x = self.embedding(indices[:, :max_len])
        output, _ = self.rnn(x)
        return self.linear(output)

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        self.eval()
        device = next(self.parameters()).device

        # 1. Кодируем префикс → добавляем BOS вручную
        prefix_ids = self.dataset.text2ids(prefix) if prefix else []
        seq = [self.dataset.bos_id] + prefix_ids

        # 2. Защита от переполнения
        if len(seq) >= self.max_length:
            res = seq[1:self.max_length]  # убрали BOS, обрезали до лимита
            if self.dataset.eos_id in res:
                res = res[:res.index(self.dataset.eos_id)]
            return self.dataset.ids2text(res)

        # 3. Инициализация скрытого состояния
        batch_size = 1
        if isinstance(self.rnn, nn.LSTM):
            h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            hidden = (h, c)
        else:
            hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

        # 4. Прогоняем префикс через модель
        input_tensor = torch.tensor([seq], device=device)  # (1, T)
        embedded = self.embedding(input_tensor)
        _, hidden = self.rnn(embedded, hidden)  # Нам нужно ТОЛЬКО скрытое состояние

        # 5. Генерация по одному токену
        generated = seq.copy()
        while len(generated) < self.max_length:
            # Текущий токен для генерации — последний в последовательности
            last_token = torch.tensor([[generated[-1]]], device=device)  # (1, 1)
            emb = self.embedding(last_token)  # (1, 1, E)
            output, hidden = self.rnn(emb, hidden)  # (1, 1, H)
            logits = self.linear(output.squeeze(1)) / temp  # (1, V)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if next_token == self.dataset.eos_id:
                break

        # 6. Постобработка: убрать BOS, обрезать до EOS
        result_ids = generated[1:]  # убрали BOS
        if self.dataset.eos_id in result_ids:
            result_ids = result_ids[:result_ids.index(self.dataset.eos_id)]
        return self.dataset.ids2text(result_ids)