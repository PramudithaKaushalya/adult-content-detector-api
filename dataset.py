import torch
from torch.utils.data import Dataset
from collections import Counter
from nltk.tokenize import word_tokenize
import re

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        # Convert text to indices
        text_indices = [self.vocab.get(token, self.vocab['<PAD>']) for token in text]
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def clean_text(text):
    """Preprocess text: remove special characters and convert to lowercase."""
    text = re.sub(r"[^අ-ෆ ]", "", text)  # Keep Sinhala characters
    return text.lower()

def tokenize_text(text):
    """Tokenize and clean text."""
    return word_tokenize(clean_text(text))

def build_vocab(data, min_freq=1):
    """Build vocabulary from tokenized data."""
    counter = Counter()
    for tokens, _ in data:  # tokens is already a list (tokenized)
        counter.update(tokens)

    vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<PAD>'] = len(vocab)  # Add padding token
    vocab['<UNK>'] = len(vocab)  # Add unknown token

    return vocab

def collate_fn(batch):
    texts, labels = zip(*batch)
    # Pad sequences to same length
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.stack(labels)
