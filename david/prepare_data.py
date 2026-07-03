"""prepare_data.py — turn raw Shakespeare text into training-ready id tensors.

The whole 'dataset pipeline' at char level (nanoGPT's prepare.py, adapted):
  1. read text  2. build the char vocab from it  3. encode text -> one long id tensor
  4. split 90/10 train/val  5. save everything to data/prepared.pt
"""
import torch

text = open("data/tinyshakespeare.txt").read()

# ---- the tokenizer IS the vocab: char -> index in the sorted unique-char list ----
chars = sorted(set(text))
vocab_size = len(chars)                                  # 65
stoi = {ch: i for i, ch in enumerate(chars)}             # encode table (string -> int)
itos = {i: ch for i, ch in enumerate(chars)}             # decode table (int -> string)

# ---- encode the entire corpus into one long tensor of ids ----
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

# ---- train/val split: last 10% held out to measure loss on UNSEEN text ----
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

torch.save(
    {"train": train_data, "val": val_data, "stoi": stoi, "itos": itos, "vocab_size": vocab_size},
    "data/prepared.pt",
)

if __name__ == "__main__":
    print(f"vocab_size: {vocab_size}")
    print(f"total ids:  {len(data):,}   train: {len(train_data):,}   val: {len(val_data):,}")
    sample = text[:32]
    ids = [stoi[ch] for ch in sample]
    back = "".join(itos[i] for i in ids)
    print(f"round-trip: {sample!r} -> {ids[:8]}... -> {back!r}")
    assert back == sample, "encode/decode must round-trip"
    print("round-trip OK, saved to data/prepared.pt")
