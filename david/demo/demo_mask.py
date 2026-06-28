"""demo_mask.py — hyper-focused: what does the causal mask DO to the score matrix?

The mask is ADDED to the n x n self-attention scores (then softmax runs).
Goal: SEE future positions get -inf, then become exactly 0 after softmax.
"""
import torch
from torch.nn.functional import softmax

# A hardcoded 4x4 score matrix (QK^T/sqrt(d) would produce this). 4 tokens.
scores = torch.tensor([
    [2.0, 1.0, 0.5, 3.0],
    [0.3, 2.0, 1.0, 0.2],
    [1.0, 0.4, 2.0, 1.5],
    [0.7, 1.2, 0.9, 2.0],
])

# The causal mask: 0 where allowed (self + past), -inf where forbidden (future).
mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)

print("1) RAW scores (every token can see every token — including the FUTURE):")
print(scores)

print("\n2) The MASK (0 = allowed, -inf = forbidden future):")
print(mask)

masked = scores + mask
print("\n3) scores + mask  (upper-right is now -inf = blindfolded from the future):")
print(masked)

print("\n4) softmax(each row)  — the -inf spots become EXACTLY 0:")
weights = softmax(masked, dim=-1)
torch.set_printoptions(precision=3, sci_mode=False)
print(weights)

print("\nRead it row by row:")
for i in range(4):
    w = [round(x, 3) for x in weights[i].tolist()]
    print(f"   token{i} attends to -> {w}   (sum={round(sum(w),3)}, future = 0)")

print("\nWithout the mask, row 0 would spread weight onto tokens 1,2,3 (the future).")
print("With the mask, token0 can ONLY attend to itself; token1 to {0,1}; etc.")
