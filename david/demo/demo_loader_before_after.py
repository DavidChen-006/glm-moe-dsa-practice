"""demo_loader_before_after.py — the DataLoader, slow: BEFORE it exists vs AFTER.

6 tiny samples. Every __getitem__ call announces itself, so you can SEE exactly
when the loader triggers the dataset, in what order, and what comes out.
"""
import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    """sample i = ([i,i,i,i], [i*10,...]). Nothing stored — cooked on demand."""

    def __len__(self):
        return 6

    def __getitem__(self, index):
        print(f"      (__getitem__ fired: cooking sample {index})")
        return torch.full((4,), index), torch.full((4,), index * 10)


ds = ToyDataset()

print("=" * 62)
print("BEFORE the loader — the dataset alone. You do everything.")
print("=" * 62)
print("\nask for ONE sample, ds[2]:")
input_ids, labels = ds[2]
print(f"   -> input_ids {input_ids.tolist()}   labels {labels.tolist()}")
print("   (the cooking line appeared only when we asked: LAZY)")

print("\ntraining needs BATCHES, so by hand you'd have to do all this:")
picked = [4, 0, 2]                                   # 1. pick indices yourself
print(f"   1. pick indices: {picked}")
cooked = [ds[i] for i in picked]                     # 2. cook each one
batch_in = torch.stack([c[0] for c in cooked])       # 3. stack into rectangles
batch_lab = torch.stack([c[1] for c in cooked])
print(f"   3. stack -> inputs {batch_in.tolist()}  shape {tuple(batch_in.shape)}")
print(f"             labels {batch_lab.tolist()}")

print()
print("=" * 62)
print("AFTER — DataLoader(ds, batch_size=3, shuffle=True) does all 3 jobs")
print("=" * 62)
torch.manual_seed(1)
loader = DataLoader(ds, batch_size=3, shuffle=True)
print("\nloader created. notice: NO cooking lines yet — still lazy.")
print("now pull on it with a for-loop:\n")
for step, (batch_in, batch_lab) in enumerate(loader, 1):
    print(f"   step {step} received: inputs {batch_in.tolist()}  shape {tuple(batch_in.shape)}")
    print(f"                   labels {batch_lab.tolist()}\n")

print("what each step did:  shuffled deck -> 3 indices -> 3 __getitem__ calls")
print("                     -> stack to (3, 4) -> hand to the loop body")
print("2 steps x 3 samples = all 6 served once = ONE EPOCH.")

print("\nloop again (epoch 2) — same 6 samples, new shuffle:")
for step, (batch_in, _) in enumerate(loader, 1):
    print(f"   step {step}: inputs {batch_in.tolist()}")
