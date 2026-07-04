"""prepare_jsonl.py — convert raw Shakespeare into minimind-style pretrain jsonl.

tinyshakespeare.txt is speeches separated by blank lines:
    First Citizen:
    Before we proceed any further, hear me speak.
    <blank>
Each speech becomes one document: {"text": "..."} per line — the exact raw shape
minimind's PretrainDataset expects (sample['text']).
"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

text = open("data/tinyshakespeare.txt").read()

speeches = [block.strip() for block in text.split("\n\n") if block.strip()]

n = int(0.9 * len(speeches))                       # 90/10 doc split, mirroring the bin split
with open("data/pretrain_shakespeare.jsonl", "w") as f:
    for speech in speeches[:n]:
        f.write(json.dumps({"text": speech}) + "\n")
with open("data/val_shakespeare.jsonl", "w") as f:
    for speech in speeches[n:]:
        f.write(json.dumps({"text": speech}) + "\n")

if __name__ == "__main__":
    lengths = [len(s) for s in speeches]
    print(f"documents written: {len(speeches):,}  (train {n:,} / val {len(speeches) - n:,})")
    print(f"chars per doc: min {min(lengths)}, median {sorted(lengths)[len(lengths)//2]}, max {max(lengths)}")
    print(f"first doc: {json.dumps({'text': speeches[0]})[:100]}...")
