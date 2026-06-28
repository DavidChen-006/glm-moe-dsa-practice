"""demo_v.py — what does V (the 'values') look like?

V is a MATRIX: one row per token, one column per dimension.
Row i = the "value vector" that token i offers up to be blended by attention.
"""
import torch
from torch.nn.functional import softmax

# 3 tokens, each described by a 4-dim value vector.
# (real attention: V = x @ Wv, but here we just hand-write it so you can SEE it.)
V = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # token 0's value vector
    [0.0, 2.0, 0.0, 0.0],   # token 1's value vector
    [0.0, 0.0, 3.0, 0.0],   # token 2's value vector
])
















print("V =")
print(V)
print()
print("type :", type(V).__name__)
print("shape:", tuple(V.shape), " -> (num_tokens, head_dim) = (3 tokens, 4 dims)")
print("so V is a MATRIX: a stack of 3 value vectors, one per token.")
print()
print("token 1's value vector (row 1):", V[1])


# ============================================================
# softmax a score matrix, then multiply it by V — with detail
# ============================================================
print("\n" + "=" * 55)
print("softmax(scores) @ V")
print("=" * 55)

scores = torch.tensor([[2.0, 1.0, 0.0],     # query 0's raw scores vs the 3 tokens
                       [0.0, 0.0, 4.0]])    # query 1's raw scores vs the 3 tokens
weights = softmax(scores, dim=-1)           # softmax along each ROW -> each row sums to 1

print("\nscores (raw):")
for i, r in enumerate(scores.tolist()):
    print(f"   query{i}: {r}")
print("\nweights = softmax(scores)  (each row sums to 1):")
for i, r in enumerate(weights.tolist()):
    print(f"   query{i}: {[round(x, 3) for x in r]}  (sum={round(sum(r), 3)})")

out = weights @ V                           # (2x3) @ (3x4) -> 2x4

print("\nDETAILED computation (weighted sum of V's rows):")
for i in range(weights.shape[0]):
    w = weights[i]
    print(f"\n  query{i}:")
    print(f"     {w[0]:.3f} * row0 {V[0].tolist()} = {[round(x,3) for x in (w[0]*V[0]).tolist()]}")
    print(f"   + {w[1]:.3f} * row1 {V[1].tolist()} = {[round(x,3) for x in (w[1]*V[1]).tolist()]}")
    print(f"   + {w[2]:.3f} * row2 {V[2].tolist()} = {[round(x,3) for x in (w[2]*V[2]).tolist()]}")
    print(f"   = {[round(x,3) for x in out[i].tolist()]}")

print("\nfinal output (2x4):")
print(out)


# ============================================================
# Q and K — where the scores actually come from (Q @ K^T)
# 3 tokens, 4 features each — so Q@K^T is 3x3, ready for softmax @ V (3x4).
# ============================================================
print("\n" + "=" * 55)
print("Q @ K^T  ->  the raw score matrix")
print("=" * 55)

Q = torch.tensor([          # each row = one token's "what I'm looking for"
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
])  # 3 x 4

K = torch.tensor([          # each row = one token's "what I offer" (name tag)
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
])  # 3 x 4

print("Q (3x4):\n", Q)
print("K (3x4):\n", K)
print("\nK^T (4x3) — transposed so the inner dims match for the multiply:\n", K.T)

scores_qk = Q @ K.T         # (3x4) @ (4x3) -> 3x3
print("\nscores = Q @ K^T  (3x3):  entry[i,j] = token i's query . token j's key")
print(scores_qk)
print("\n-> this 3x3 is exactly what would go into  softmax(scores/sqrt(d) + mask) @ V.")
