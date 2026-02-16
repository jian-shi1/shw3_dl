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
        L = lengths.max().item()
        x = self.embedding(indices[:, :L])
        output, _ = self.rnn(x)
        return self.linear(output)

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        self.eval()
        device = next(self.parameters()).device

        # Кодируем префикс → добавляем BOS
        prefix_ids = self.dataset.text2ids(prefix) if prefix else []
        generated = [self.dataset.bos_id] + prefix_ids

        # Защита от переполнения
        if len(generated) >= self.max_length:
            # Убираем BOS и всё после EOS (если есть)
            res = generated[1:]  # убрали BOS
            if self.dataset.eos_id in res:
                res = res[:res.index(self.dataset.eos_id)]
            return self.dataset.ids2text(res)

        # Подготовка начального скрытого состояния
        batch_size = 1
        if isinstance(self.rnn, nn.LSTM):
            h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            hidden = (h, c)
        else:
            hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

        # Прогоняем префикс
        input_tensor = torch.tensor([generated], device=device)
        embedded = self.embedding(input_tensor)
        output, hidden = self.rnn(embedded, hidden)

        # Генерация по одному токену
        while len(generated) < self.max_length:
            logits = self.linear(output[:, -1, :]) / temp  # (1, V)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == self.dataset.eos_id:
                break

            # Следующий шаг
            next_input = torch.tensor([[next_token]], device=device)
            next_embedded = self.embedding(next_input)
            output, hidden = self.rnn(next_embedded, hidden)

        # Финальная постобработка
        result = generated[1:]  # убираем BOS
        if self.dataset.eos_id in result:
            result = result[:result.index(self.dataset.eos_id)]
        return self.dataset.ids2text(result)