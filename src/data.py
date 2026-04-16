import json
import os
import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """Character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos.get(t, "") for t in tokens)

    def to_dict(self) -> dict:
        return {"vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        tok = cls.__new__(cls)
        tok.vocab = d["vocab"]
        tok.vocab_size = len(tok.vocab)
        tok.stoi = {c: i for i, c in enumerate(tok.vocab)}
        tok.itos = {i: c for i, c in enumerate(tok.vocab)}
        return tok

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class TextDataset(Dataset):
    """Dataset that yields (input, target) pairs of fixed length."""

    def __init__(self, data: list[int], block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


def prepare_data(config: dict):
    """Load text, build tokenizer, split into train/val datasets."""
    path = config["data"]["dataset_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it with:\n"
            "  curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    with open(path, encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    split = int(config["data"]["train_split"] * len(data))
    block_size = config["model"]["block_size"]

    train_dataset = TextDataset(data[:split], block_size)
    val_dataset = TextDataset(data[split:], block_size)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train tokens: {split:,}  |  Val tokens: {len(data) - split:,}")

    return tokenizer, train_dataset, val_dataset
