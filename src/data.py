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
        return {"type": "char", "vocab": self.vocab}

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
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class BPETokenizer:
    """Byte Pair Encoding tokenizer trained from scratch."""

    def __init__(self, text: str = None, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, str] = {}
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab_size: int = 0
        if text is not None:
            self._train(text)

    def _get_pair_counts(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for a, b in zip(tokens, tokens[1:]):
            pair = (a, b)
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def _train(self, text: str):
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(self.vocab)

        tokens = [self.stoi[c] for c in text]
        num_merges = max(0, self.target_vocab_size - len(chars))

        for rank in range(num_merges):
            pairs = self._get_pair_counts(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_id = len(self.vocab)
            new_token = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_token
            self.itos[new_id] = new_token
            self.stoi[new_token] = new_id
            self.merges[best_pair] = new_id
            self.merge_ranks[best_pair] = rank
            tokens = self._merge_tokens(tokens, best_pair, new_id)

            if (rank + 1) % 100 == 0 or rank == num_merges - 1:
                print(f"\r  BPE merges: {rank + 1}/{num_merges}", end="", flush=True)

        print()
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        tokens = [self.stoi[c] for c in text if c in self.stoi]
        while len(tokens) >= 2:
            # Find the highest-priority (earliest-learned) applicable merge
            best_pair = None
            best_rank = len(self.merges)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            tokens = self._merge_tokens(tokens, best_pair, self.merges[best_pair])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocab.get(t, "") for t in tokens)

    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "vocab": [[k, v] for k, v in self.vocab.items()],
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls.__new__(cls)
        tok.target_vocab_size = d["target_vocab_size"]
        tok.vocab = {int(k): v for k, v in d["vocab"]}
        tok.itos = dict(tok.vocab)
        tok.stoi = {v: int(k) for k, v in d["vocab"]}
        tok.merges = {}
        tok.merge_ranks = {}
        for rank, (a, b, c) in enumerate(d["merges"]):
            tok.merges[(a, b)] = c
            tok.merge_ranks[(a, b)] = rank
        tok.vocab_size = len(tok.vocab)
        return tok

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def load_tokenizer(path: str) -> "CharTokenizer | BPETokenizer":
    """Load a tokenizer from JSON, auto-detecting the type."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    tok_type = d.get("type", "char")
    if tok_type == "bpe":
        return BPETokenizer.from_dict(d)
    return CharTokenizer.from_dict(d)


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

    tok_type = config["data"].get("tokenizer_type", "char")
    if tok_type == "bpe":
        bpe_vocab_size = config["data"].get("bpe_vocab_size", 1000)
        print(f"Training BPE tokenizer (target vocab: {bpe_vocab_size})...")
        tokenizer = BPETokenizer(text, vocab_size=bpe_vocab_size)
        print("Encoding corpus with BPE...")
    else:
        tokenizer = CharTokenizer(text)

    data = tokenizer.encode(text)

    split = int(config["data"]["train_split"] * len(data))
    block_size = config["model"]["block_size"]

    train_dataset = TextDataset(data[:split], block_size)
    val_dataset = TextDataset(data[split:], block_size)

    print(f"Tokenizer: {tok_type} | Vocab size: {tokenizer.vocab_size}")
    print(f"Train tokens: {split:,}  |  Val tokens: {len(data) - split:,}")

    return tokenizer, train_dataset, val_dataset
