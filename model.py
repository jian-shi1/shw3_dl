import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        # Создание слоёв
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
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, input length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
        """
        # Эмбеддинг последовательности
        embedded = self.embedding(indices)  # (batch_size, seq_len, embed_size)
        
        # Сортировка для корректной упаковки
        lengths = lengths.cpu()
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        embedded_sorted = embedded[sorted_idx]
        
        # Упаковка, проход через RNN, распаковка
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
        )
        output_packed, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output_packed, batch_first=True, total_length=indices.size(1)
        )
        
        # Восстановление исходного порядка
        _, original_idx = torch.sort(sorted_idx)
        output = output[original_idx]
        
        # Линейный слой -> логиты
        logits = self.linear(output)  # (batch_size, seq_len, vocab_size)
        
        # Обрезка до максимальной длины в батче (требуется тестами)
        max_len_in_batch = lengths.max().item()
        logits = logits[:, :max_len_in_batch, :]
        
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Кодирование префикса (без спецсимволов)
        prefix_ids = self.dataset.text2ids(prefix) if prefix else []
        generated_ids = [self.dataset.bos_id] + prefix_ids
        
        # Защита от переполнения длины
        if len(generated_ids) >= self.max_length:
            generated_ids = generated_ids[:self.max_length]
            # Очистка от спецсимволов перед декодированием
            if generated_ids and generated_ids[0] == self.dataset.bos_id:
                generated_ids = generated_ids[1:]
            if self.dataset.eos_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(self.dataset.eos_id)]
            return self.dataset.ids2text(generated_ids)
        
        # Обработка префикса через модель для получения скрытого состояния
        input_tensor = torch.tensor([generated_ids], device=device)
        embedded = self.embedding(input_tensor)
        
        # Инициализация скрытого состояния
        batch_size = 1
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            c0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
        
        output, hidden = self.rnn(embedded, hidden)  # output: (1, seq_len, hidden_size)
        
        # Генерация токенов
        current_length = len(generated_ids)
        while current_length < self.max_length:
            # Получаем логиты для следующего токена (последний шаг выхода RNN)
            logits = self.linear(output[:, -1, :])  # (1, vocab_size)
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token_id)
            current_length += 1
            
            # Прерывание при генерации EOS или достижении лимита
            if next_token_id == self.dataset.eos_id or current_length >= self.max_length:
                break
            
            # Подготовка следующего шага: подаём сгенерированный токен в RNN
            next_input = torch.tensor([[next_token_id]], device=device)
            next_embedded = self.embedding(next_input)
            output, hidden = self.rnn(next_embedded, hidden)  # output: (1, 1, hidden_size)
        
        # Постобработка: удаление BOS, обрезка до EOS
        if generated_ids and generated_ids[0] == self.dataset.bos_id:
            generated_ids = generated_ids[1:]
        if self.dataset.eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(self.dataset.eos_id)]
        
        return self.dataset.ids2text(generated_ids)
