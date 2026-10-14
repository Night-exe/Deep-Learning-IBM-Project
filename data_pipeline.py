"""
Stage 1: Data pipeline for English -> French Neural Machine Translation
Project: Seq2Seq with Attention

DATASET
-------
This uses the Tatoeba-derived English-French sentence pairs distributed at:
    https://www.manythings.org/anki/fra-eng.zip
(the same dataset used in PyTorch's official seq2seq-with-attention tutorial)

To use the real dataset:
    1. Download https://www.manythings.org/anki/fra-eng.zip
    2. Unzip it -> you'll get 'fra.txt' (format: english<TAB>french<TAB>attribution)
    3. Place 'fra.txt' in the same folder as this script (or update DATA_PATH below)

If 'fra.txt' is not found, this script automatically falls back to a small
embedded demo corpus so the whole pipeline is runnable and testable immediately.

NOTE ON ACCENTS: sentences are stripped down to plain ASCII (accents removed,
e.g. "etait" instead of "était"). This mirrors the standard tutorial approach
and keeps the vocabulary small and simple for a first pass. Swap in a lighter
cleaning function later if you want to preserve French diacritics.

NOTE ON DIGITS: digits (0-9) are preserved during cleaning. Numbers carry
real information ("I have 5 apples" != "I have apples"), so stripping them
would remove content the target translation depends on. Two residual
limitations worth knowing:
  1. Decimal points get mangled: "3.5" becomes the tokens "3" and ".5"
     (the period sticks to the following digit) because punctuation
     splitting runs before digit-aware handling. Fine for whole numbers,
     messy for decimals - a decimal-aware regex would be a further
     improvement if your data has prices/measurements.
  2. This is still word-level vocabulary, not subword/character-level, so a
     number the model never saw during training (e.g. "5" was in the
     training set but "247" was not) will map to <unk> at inference time.
     A production system would use subword tokenization (e.g. BPE) to
     generalize to unseen numbers; out of scope for this project.
"""

import re
import unicodedata
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "fra.txt"      # tab-separated english<TAB>french lines
MAX_LEN = 12                # keep sentences short, per project spec
MIN_LEN = 2
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 42

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

random.seed(SEED)


# ----------------------------
# Text cleaning
# ----------------------------
def unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_sentence(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Load pairs (real dataset, or embedded demo fallback)
# ----------------------------
DEMO_PAIRS = [
    ("go.", "va !"),
    ("run!", "cours !"),
    ("i am cold.", "j'ai froid."),
    ("i am fine.", "je vais bien."),
    ("she is reading a book.", "elle lit un livre."),
    ("the cat sat on the mat.", "le chat etait assis sur le tapis."),
    ("he plays football every sunday.", "il joue au football tous les dimanches."),
    ("we are going to the market.", "nous allons au marche."),
    ("i love learning new languages.", "j'aime apprendre de nouvelles langues."),
    ("this is a small example dataset.", "ceci est un petit ensemble de donnees d'exemple."),
    ("what time is it.", "quelle heure est il."),
    ("i am very happy today.", "je suis tres heureux aujourd'hui."),
    ("can you help me please.", "peux tu m'aider s'il te plait."),
    ("the weather is nice today.", "il fait beau aujourd'hui."),
    ("she speaks french very well.", "elle parle tres bien francais."),
    ("i have 5 apples.", "j'ai 5 pommes."),
]


def load_pairs(path: str):
    file = Path(path)
    if file.exists():
        pairs = []
        with open(file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                eng, fra = parts[0], parts[1]
                pairs.append((normalize_sentence(eng), normalize_sentence(fra)))
        print(f"Loaded {len(pairs)} pairs from {path}")
    else:
        pairs = [(normalize_sentence(e), normalize_sentence(f)) for e, f in DEMO_PAIRS]
        print(
            f"'{path}' not found - using {len(pairs)} embedded demo pairs instead.\n"
            f"Download the real corpus from https://www.manythings.org/anki/fra-eng.zip "
            f"and place 'fra.txt' next to this script for real training."
        )
    return pairs


def filter_pairs(pairs):
    filtered = []
    for eng, fra in pairs:
        e_len, f_len = len(eng.split()), len(fra.split())
        if MIN_LEN <= e_len <= MAX_LEN and MIN_LEN <= f_len <= MAX_LEN:
            filtered.append((eng, fra))
    return filtered


# ----------------------------
# Vocabulary
# ----------------------------
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.word_count = {}

    def add_sentence(self, sentence: str):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence: str):
        return (
            [self.word2idx[SOS_TOKEN]]
            + [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in sentence.split()]
            + [self.word2idx[EOS_TOKEN]]
        )

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if word == EOS_TOKEN:
                break
            if word not in (SOS_TOKEN, PAD_TOKEN):
                words.append(word)
        return " ".join(words)


def build_vocabularies(pairs):
    src_vocab = Vocabulary("eng")
    tgt_vocab = Vocabulary("fra")
    for eng, fra in pairs:
        src_vocab.add_sentence(eng)
        tgt_vocab.add_sentence(fra)
    return src_vocab, tgt_vocab


# ----------------------------
# PyTorch Dataset
# ----------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng, fra = self.pairs[idx]
        src_ids = self.src_vocab.encode(eng)
        tgt_ids = self.tgt_vocab.encode(fra)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    src_padded = torch.zeros(len(batch), max(src_lens), dtype=torch.long)
    tgt_padded = torch.zeros(len(batch), max(tgt_lens), dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t

    return src_padded, tgt_padded, torch.tensor(src_lens), torch.tensor(tgt_lens)


# ----------------------------
# Build everything
# ----------------------------
def prepare_data(batch_size=32, data_path=None):
    # data_path lets you point at the real dataset from wherever it lives
    # (e.g. a Drive-mounted folder in Colab) without editing this file.
    # Falls back to the DATA_PATH constant above if not given.
    resolved_path = data_path if data_path is not None else DATA_PATH
    pairs = load_pairs(resolved_path)
    pairs = filter_pairs(pairs)

    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=VAL_SPLIT + TEST_SPLIT, random_state=SEED
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=SEED
    )

    src_vocab, tgt_vocab = build_vocabularies(train_pairs)

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    test_ds = TranslationDataset(test_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Source vocab size: {len(src_vocab)} | Target vocab size: {len(tgt_vocab)}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_data(batch_size=4)

    # quick sanity check
    src_batch, tgt_batch, src_lens, tgt_lens = next(iter(train_loader))
    print("Sample source batch shape:", src_batch.shape)
    print("Sample target batch shape:", tgt_batch.shape)
    print(
        "Decoded sample:",
        src_vocab.decode(src_batch[0].tolist()),
        "->",
        tgt_vocab.decode(tgt_batch[0].tolist()),
    )
