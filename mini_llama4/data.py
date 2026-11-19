"""TinyShakespeare + a character-level tokenizer."""

import urllib.request
from pathlib import Path

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


def load_text(path: Path = DATA_PATH) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class Dataset:
    def __init__(self, text: str, tokenizer: CharTokenizer, val_frac: float = 0.1):
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.splits = {"train": ids[:n], "val": ids[n:]}

    def get_batch(self, split: str, batch_size: int, seq_len: int, device="cpu"):
        """x = tokens[i : i+T], y = tokens[i+1 : i+T+1]  (y is x shifted by one: next-token targets)."""
        data = self.splits[split]
        starts = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s : s + seq_len] for s in starts])
        y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        return x.to(device), y.to(device)
