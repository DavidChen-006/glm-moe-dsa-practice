"""A/B: loading jsonl with vs without PretrainDataset.

Run from repo root:
    python david/demo/demo_pretrain_dataset.py
    python david/demo/demo_pretrain_dataset.py --max-length 64
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DAVID_DIR = os.path.dirname(DEMO_DIR)
TRAINING_DIR = os.path.join(DAVID_DIR, "training")
DATA_PATH = os.path.join(TRAINING_DIR, "data/pretrain_shakespeare.jsonl")

sys.path.insert(0, TRAINING_DIR)
from lm_dataset import PretrainDataset  # noqa: E402


def load_without_pretrain_dataset(data_path, index=0):
    """A: raw jsonl — you get text, not training-ready tensors."""
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(index)


def show_a(data_path, index):
    print("=" * 60)
    print("A) WITHOUT PretrainDataset — raw jsonl")
    print("=" * 60)

    sample = load_without_pretrain_dataset(data_path, index)
    print(f"\nWhat you get from the file (sample {index}):")
    print(f"  type:  {type(sample).__name__}")
    print(f"  keys:  {list(sample.keys())}")
    print(f"  text:  {sample['text'][:80]!r}...")

    print("\nWhat is missing for training:")
    print("  - not tokenized (still a string, not token IDs)")
    print("  - no fixed length (can't batch with DataLoader)")
    print("  - no BOS / EOS framing")
    print("  - no padding or pad masking in labels")
    print("  - not PyTorch tensors")

    # show naive tokenize — still not training-ready
    tok = AutoTokenizer.from_pretrained(DAVID_DIR)
    naive_ids = tok(sample["text"]).input_ids
    print(f"\nNaive tokenizer call alone still gives variable length: {len(naive_ids)} tokens")
    print(f"  first 12 ids: {naive_ids[:12]}")


def show_b(data_path, tok, max_length, index):
    print("\n" + "=" * 60)
    print("B) WITH PretrainDataset — training-ready tensors")
    print("=" * 60)

    ds = PretrainDataset(data_path, tok, max_length=max_length)
    input_ids, labels = ds[index]

    n_real = (input_ids != tok.pad_token_id).sum().item()
    n_pad = max_length - n_real

    print(f"\nWhat you get from ds[{index}]:")
    print(f"  input_ids  type={type(input_ids).__name__}  shape={tuple(input_ids.shape)}")
    print(f"  labels     type={type(labels).__name__}  shape={tuple(labels.shape)}")
    print(f"  real tokens: {n_real}  padding: {n_pad}")

    print("\nPretrainDataset handled for you:")
    print(f"  BOS at start:     id {input_ids[0].item()} (bos_token_id={tok.bos_token_id})")
    print(f"  EOS before pad:   id {input_ids[n_real - 1].item()} (eos_token_id={tok.eos_token_id})")
    print(f"  fixed length:     always {max_length} (truncate + pad)")
    print(f"  labels on pad:    -100 so loss ignores padding")

    print(f"\nDecoded start: {tok.decode(input_ids[:20].tolist())!r}")
    print(f"Last real token decoded: {tok.decode([input_ids[n_real - 1].item()])!r}")

    # show DataLoader can batch because lengths match
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch_input_ids, batch_labels = next(iter(loader))
    print(f"\nDataLoader batch (works because every item is length {max_length}):")
    print(f"  input_ids shape: {tuple(batch_input_ids.shape)}")
    print(f"  labels    shape: {tuple(batch_labels.shape)}")


def main():
    parser = argparse.ArgumentParser(description="A/B: PretrainDataset vs raw jsonl")
    parser.add_argument("--max-length", type=int, default=64, help="shorter for readable demo output")
    parser.add_argument("--index", type=int, default=0, help="which jsonl document to inspect")
    args = parser.parse_args()

    if not os.path.isfile(DATA_PATH):
        print(f"Missing {DATA_PATH}")
        print("Run from repo root after jsonl data exists.")
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(DAVID_DIR)

    show_a(DATA_PATH, args.index)
    show_b(DATA_PATH, tok, args.max_length, args.index)

    print("\n" + "=" * 60)
    print("Takeaway: PretrainDataset turns raw jsonl text into fixed-length")
    print("(input_ids, labels) tensors — ready for DataLoader and the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
