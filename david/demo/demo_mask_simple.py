"""demo_mask_simple.py — the simplest possible mask: 6 tokens, hide the 6th.

No attention math yet. Just SEE the variables: the tokens, and a mask that
covers the last token (the 'answer' we want the model to predict).
"""
import torch

# 6 tokens (just words so we can read them)
tokens = ["The", "cat", "sat", "on", "the", "mat"]
print("tokens:", tokens)
print("indices:", list(range(len(tokens))))

# A mask, one entry per token:
#   0     = visible  (the model is allowed to look at this token)
#   -inf  = hidden   (covered up — the model must NOT see this)
# Here we cover ONLY the 6th token (index 5) — that's the "answer".
mask = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, float("-inf")])

print("\nmask:", mask.tolist())
print("       ", "  ".join(f"{t}" for t in tokens))

print("\nWhat the model can SEE vs what's HIDDEN:")
for i, (t, m) in enumerate(zip(tokens, mask.tolist())):
    state = "VISIBLE" if m == 0.0 else "HIDDEN (the answer)"
    print(f"   token{i}  '{t}'  mask={m:>5}  -> {state}")

print("\nSo: the model sees 'The cat sat on the ___' and must predict the hidden 'mat'.")
