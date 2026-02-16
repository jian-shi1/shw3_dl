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

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(embed_size, hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, input length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        # Получаем эмбеддинги
        embeddings = self.embedding(indices)  # (batch_size, max_length, embed_size)
        
        # Пропускаем через RNN
        rnn_out, _ = self.rnn(embeddings)  # (batch_size, max_length, hidden_size)
        
        # Применяем линейный слой для получения логитов
        logits = self.linear(rnn_out)  # (batch_size, max_length, vocab_size)
        
        # Обрезаем выход до максимальной длины в батче
        max_len = lengths.max().item()
        logits = logits[:, :max_len, :]
        
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
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        device = next(self.parameters()).device
        
        # Кодируем префикс
        if prefix:
            prefix_ids = self.dataset.text2ids(prefix)
            # Добавляем BOS в начало
            input_ids = [self.dataset.bos_id] + prefix_ids
        else:
            # Если префикс пустой, начинаем только с BOS
            input_ids = [self.dataset.bos_id]
        
        # Инициализируем скрытое состояние
        hidden = None
        
        # Прогоняем префикс через модель для накопления скрытого состояния
        for token_id in input_ids:
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
            embedding = self.embedding(token_tensor)  # (1, 1, embed_size)
            
            if isinstance(self.rnn, nn.LSTM):
                rnn_out, hidden = self.rnn(embedding, hidden)
            else:
                rnn_out, hidden = self.rnn(embedding, hidden)
        
        # Генерируем новые токены
        generated_ids = input_ids.copy()
        
        while len(generated_ids) < self.max_length:
            # Получаем логиты для следующего токена
            logits = self.linear(rnn_out)  # (1, 1, vocab_size)
            logits = logits.squeeze() / temp  # (vocab_size,)
            
            # Семплируем следующий токен
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Проверяем на EOS
            if next_token == self.dataset.eos_id:
                break
            
            generated_ids.append(next_token)
            
            # Обновляем скрытое состояние для следующего шага
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            embedding = self.embedding(token_tensor)
            
            if isinstance(self.rnn, nn.LSTM):
                rnn_out, hidden = self.rnn(embedding, hidden)
            else:
                rnn_out, hidden = self.rnn(embedding, hidden)
        
        # Декодируем в текст (убираем BOS из начала)
        if prefix:
            # Возвращаем префикс + сгенерированное
            generated_text = self.dataset.ids2text(generated_ids[1:])
        else:
            # Если префикса не было, возвращаем все кроме BOS
            generated_text = self.dataset.ids2text(generated_ids[1:])
        
        return generated_text