"""train.py — pretrain the toy GLM on tiny-shakespeare.

Mirrors minimind/train_pretrain.py (structure, names, flow), minus the multi-GPU
armor (DDP/DistributedSampler) since this runs on one MacBook. Data comes from
data/train.bin + val.bin (prepare_data.py) instead of their jsonl-at-runtime path.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modeling_glm_moe_dsa import GlmMoeDsaConfig


class PretrainDataset(Dataset):
    """Mirror of minimind's PretrainDataset, over a pre-tokenized .bin instead of jsonl.
    Sample i = window of ids starting at i; input/labels = the window shifted by one."""

    def __init__(self, data_path, max_length=256):
        self.ids = np.fromfile(data_path, dtype=np.uint16)
        self.max_length = max_length

    def __len__(self):
        return len(self.ids) - self.max_length - 1

    def __getitem__(self, index):
        window = self.ids[index : index + self.max_length + 1].astype(np.int64)
        input_ids = torch.from_numpy(window[:-1])   # tokens 0..255
        labels = torch.from_numpy(window[1:])       # tokens 1..256  (the "next token" answers)
        return input_ids, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy GLM Pretraining")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str,
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--data_path", type=str, default="data/train.bin")
    args = parser.parse_args()

    # ========== 1. seed ==========
    torch.manual_seed(42)

    # ========== 2. dirs + model config ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = GlmMoeDsaConfig(
        vocab_size=6400,                       # must match tokenizer.json
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 2,
    )
    lm_config._attn_implementation = "eager"

    # ========== 5. data (model + optimizer arrive in the next spike) ==========
    train_ds = PretrainDataset(args.data_path, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # spike-1 sanity check: one batch, shapes + the shift visible
    input_ids, labels = next(iter(loader))
    print(f"dataset windows: {len(train_ds):,}")
    print(f"batch input_ids: {tuple(input_ids.shape)}  labels: {tuple(labels.shape)}")
    print(f"shift check: input[0,1:6] = {input_ids[0, 1:6].tolist()}")
    print(f"             labels[0,0:5] = {labels[0, 0:5].tolist()}   <- must be identical")
