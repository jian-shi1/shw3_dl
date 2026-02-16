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
        max_len_in_batch = int(lengths.max().item())
    
        # Обрезаем вход до максимальной длины в батче
        input_seq = indices[:, :max_len_in_batch]  # (B, L_actual)
        
        embedded = self.embedding(input_seq)       # (B, L_actual, E)
        output, _ = self.rnn(embedded)             # (B, L_actual, H)
        logits = self.linear(output)               # (B, L_actual, V)
        
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
        generated = [self.dataset.bos_id] + prefix_ids
        
        # Защита от переполнения длины
        if len(generated_ids) >= self.max_length:
            generated_ids = generated_ids[:self.max_length]
            # Очистка от спецсимволов перед декодированием
            if generated_ids and generated_ids[0] == self.dataset.bos_id:
                generated_ids = generated_ids[1:]
            if self.dataset.eos_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(self.dataset.eos_id)]
            return self.dataset.ids2text(generated_ids)
        
        # Инициализация скрытого состояния
        batch_size = 1
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            c0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
        
        input_tensor = torch.tensor([generated], device=device)  # (1, T)
        embedded = self.embedding(input_tensor)  # (1, T, E)
        output, hidden = self.rnn(embedded, hidden)  # output: (1, T, H)
        
        # Генерация по одному токену
        while len(generated) < self.max_length:
            last_hidden = output[:, -1:, :]  # (1, 1, H)
            logits = self.linear(last_hidden)  # (1, 1, V)
            logits = logits.squeeze(1) / temp  # (1, V)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            if next_token == self.dataset.eos_id:
                break
            
            # Подготовка следующего шага
            next_input = torch.tensor([[next_token]], device=device)  # (1, 1)
            next_embedded = self.embedding(next_input)  # (1, 1, E)
            output, hidden = self.rnn(next_embedded, hidden)  # (1, 1, H)
        
        # Постобработка: убрать BOS, обрезать до EOS
        result_ids = generated
        if result_ids and result_ids[0] == self.dataset.bos_id:
            result_ids = result_ids[1:]
        if self.dataset.eos_id in result_ids:
            result_ids = result_ids[:result_ids.index(self.dataset.eos_id)]
        
        return self.dataset.ids2text(result_ids)
