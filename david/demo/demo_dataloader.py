"""Playground for DataLoader — see how Dataset (1 item) becomes batches.

Try these:
    python david/demo/demo_dataloader.py
    python david/demo/demo_dataloader.py --batch-size 2
    python david/demo/demo_dataloader.py --no-shuffle
"""

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

SEQ_LEN = 4  # fixed length so DataLoader can stack batches (like train.py's max_seq_len)


class ToyDataset(Dataset):
    """Five tiny examples — like PretrainDataset, but with numbers instead of tokens."""

    def __init__(self):
        # each example: input_ids and labels, both length SEQ_LEN
        self.examples = [
            (torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4, 5])),
            (torch.tensor([10, 20, 30, 40]), torch.tensor([20, 30, 40, 50])),
            (torch.tensor([7, 8, 9, 0]), torch.tensor([8, 9, 0, 1])),
            (torch.tensor([100, 200, 300, 400]), torch.tensor([200, 300, 400, 500])),
            (torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1])),
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def show_one_item(ds):
    input_ids, labels = ds[0]
    print("1. One item from Dataset (index 0):")
    print(f"   input_ids = {input_ids.tolist()}  shape {tuple(input_ids.shape)}")
    print(f"   labels    = {labels.tolist()}  shape {tuple(labels.shape)}")


def show_batches(loader, shuffle, max_batches=3):
    print(f"\n2. Batches from DataLoader (batch_size={loader.batch_size}, shuffle={shuffle}):")
    for step, (input_ids, labels) in enumerate(loader, start=1):
        print(f"   batch {step}:")
        print(f"     input_ids shape = {tuple(input_ids.shape)}  values = {input_ids.tolist()}")
        print(f"     labels    shape = {tuple(labels.shape)}  values = {labels.tolist()}")
        if step >= max_batches:
            remaining = len(loader) - max_batches
            if remaining > 0:
                print(f"   ... {remaining} more batch(es) not shown")
            break


def show_shuffle_effect(batch_size):
    ds = ToyDataset()
    print(f"\n3. Shuffle changes order between epochs (batch_size={batch_size}):")
    for epoch in range(2):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        first_batch = next(iter(loader))[0][:, 0].tolist()  # first token of each example in batch 1
        print(f"   epoch {epoch + 1}, batch 1 first tokens: {first_batch}")


def main():
    parser = argparse.ArgumentParser(description="Tiny DataLoader demo")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-shuffle", action="store_true", help="disable shuffling")
    args = parser.parse_args()

    ds = ToyDataset()
    shuffle = not args.no_shuffle
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    print(f"Dataset length: {len(ds)} examples")
    print(f"DataLoader length: {len(loader)} batches per epoch\n")

    show_one_item(ds)
    show_batches(loader, shuffle)

    if shuffle:
        show_shuffle_effect(args.batch_size)

    print("\nTakeaway: Dataset returns one (input, label) pair; DataLoader stacks them into batches.")


if __name__ == "__main__":
    main()
