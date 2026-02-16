import torch
from torch import nn
from typing import Type
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Языковая модель на основе RNN/LSTM.
        """
        super().__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

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

        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.
        Возвращает логиты следующих токенов для каждой позиции.
        """
        # обрезаем последовательности по их максимальной длине в батче
        max_len_in_batch = int(lengths.max().item())
        indices = indices[:, :max_len_in_batch]  # (B, L)

        embedded = self.embedding(indices)       # (B, L, E)
        output, _ = self.rnn(embedded)           # (B, L, H)
        logits = self.linear(output)             # (B, L, V)

        return logits  # форма: (B, L, vocab_size)

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.0) -> str:
        """
        Генерация нового текста по префиксу.
        """
        self.eval()
        device = next(self.parameters()).device

        # перекодируем префикс → индексы без спецсимволов
        prefix_ids = self.dataset.text2ids(prefix) if prefix else []
        generated = [self.dataset.bos_id] + prefix_ids
        if len(generated) >= self.max_length:
            generated = generated[:self.max_length]

        # инициализация скрытого состояния
        batch_size = 1
        if isinstance(self.rnn, nn.LSTM):
            h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            hidden = (h, c)
        else:
            hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

        # прогоняем уже известные токены
        input_tensor = torch.tensor([generated], device=device)
        embedded = self.embedding(input_tensor)
        output, hidden = self.rnn(embedded, hidden)

        # autoregressive generation
        while len(generated) < self.max_length:
            last_output = output[:, -1:, :]               # (1, 1, H)
            logits = self.linear(last_output).squeeze(1)  # (1, V)
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            if next_token == self.dataset.eos_id:
                break

            # подаём токен дальше без пересчёта всей последовательности
            next_input = torch.tensor([[next_token]], device=device)
            next_embedded = self.embedding(next_input)
            output, hidden = self.rnn(next_embedded, hidden)

        # убираем BOS и всё после EOS
        gen_ids = generated
        if gen_ids and gen_ids[0] == self.dataset.bos_id:
            gen_ids = gen_ids[1:]
        if self.dataset.eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(self.dataset.eos_id)]

        text = self.dataset.ids2text(gen_ids)
        # гарантируем, что возвращённая строка начинается с prefix
        if not text.startswith(prefix):
            text = prefix + text
        return text