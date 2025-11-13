import numpy as np
import torch
import torch.nn as nn


class LinearProbeModule(nn.Module):
    def __init__(self, embedding_dim=4096, num_labels=2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query)

        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, hidden, attention_mask=None):
        hidden = hidden.float()

        attention_scores = torch.matmul(hidden, self.query.transpose(-1, -2)).squeeze(
            -1
        )  # (batch_size, seq_length)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, -1e9
            )  # Numerical stability

        attention_weights = torch.nn.functional.softmax(
            attention_scores, dim=-1
        )  # (batch_size, seq_length)

        context_vector = torch.sum(
            hidden * attention_weights.unsqueeze(-1), dim=1
        )  # (batch_size, hidden_size)

        logits = self.classifier(context_vector)  # (batch_size, num_labels)

        return logits
