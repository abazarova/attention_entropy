import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class LinearProbeDataset(Dataset):
    def __init__(self, hiddens, targets):
        self.hiddens = hiddens
        self.targets = targets

    def __getitem__(self, idx):
        return self.hiddens[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    
    @staticmethod
    def collate_fn(batch):
        # Separate the sequences and labels
        sequences, labels = zip(*batch)
        
        # Remove the first dimension (batch size of 1) from each sequence
        sequences = [seq for seq in sequences]

        # Pad the sequences
        padded_sequences = pad_sequence(sequences, batch_first=True)
        
        # Generate attention mask
        attention_mask = torch.tensor([[1]*len(seq) + [0]*(padded_sequences.shape[1] - len(seq)) for seq in sequences])
        
        # Convert labels to tensor
        labels = torch.tensor(labels)
        
        return {'hidden': padded_sequences, 'attention_mask': attention_mask, 'target': labels}