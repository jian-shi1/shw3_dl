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
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with packed sequences.
        Returns logits of shape (batch_size, max_length_in_batch, vocab_size).
        """
        embedded = self.embedding(indices) # (B, T, E)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # (B, L, H)
        logits = self.linear(output) # (B, L, V)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        self.eval()
        device = next(self.parameters()).device

        prefix_ids = self.dataset.text2ids(prefix) if prefix else []
        generated = [self.dataset.bos_id] + prefix_ids

        if len(generated) >= self.max_length:
            generated = generated[:self.max_length]
            if generated and generated[0] == self.dataset.bos_id:
                generated = generated[1:]
            if self.dataset.eos_id in generated:
                generated = generated[:generated.index(self.dataset.eos_id)]
            return self.dataset.ids2text(generated)

        batch_size = 1
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            c0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)

        input_tensor = torch.tensor([generated], device=device)
        embedded = self.embedding(input_tensor)
        output, hidden = self.rnn(embedded, hidden)

        while len(generated) < self.max_length:
            last_hidden = output[:, -1:, :]
            logits = self.linear(last_hidden)
            logits = logits.squeeze(1) / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            if next_token == self.dataset.eos_id:
                break

            next_input = torch.tensor([[next_token]], device=device)
            next_embedded = self.embedding(next_input)
            output, hidden = self.rnn(next_embedded, hidden)

        result_ids = generated
        if result_ids and result_ids[0] == self.dataset.bos_id:
            result_ids = result_ids[1:]
        if self.dataset.eos_id in result_ids:
            result_ids = result_ids[:result_ids.index(self.dataset.eos_id)]

        return self.dataset.ids2text(result_ids)