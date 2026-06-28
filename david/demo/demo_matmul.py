"""demo_matmul.py — SEE what weights · V actually computes (two ways).

weights (1x3) = the recipe   |   V (3x4) = the ingredients (value vectors)
"""
import torch

weights = torch.tensor([[0.66, 0.24, 0.10]])      # 1 x 3  (one query's recipe, sums to 1)
V = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # row0 = token 0's value vector
    [0.0, 2.0, 0.0, 0.0],   # row1 = token 1's value vector
    [0.0, 0.0, 3.0, 0.0],   # row2 = token 2's value vector
])                                                # 3 x 4

out = weights @ V                                 # 1 x 4

print("weights (1x3):", weights.tolist())
print("V (3x4):")
for i, r in enumerate(V.tolist()):
    print(f"   row{i} = {r}")
print()

print("VIEW 1 — weighted SUM of V's rows (the attention way):")
print(f"   out = {weights[0,0]:.2f}*row0 + {weights[0,1]:.2f}*row1 + {weights[0,2]:.2f}*row2")
print(f"       = {(weights[0,0]*V[0]).tolist()}")
print(f"       + {(weights[0,1]*V[1]).tolist()}")
print(f"       + {(weights[0,2]*V[2]).tolist()}")
print(f"       = {out[0].tolist()}")
print()

print("VIEW 2 — each OUTPUT cell = row-of-weights DOT column-of-V:")
for j in range(V.shape[1]):
    col = V[:, j]
    dot = (weights[0] * col).sum()
    print(f"   out[{j}] = [0.66,0.24,0.10] . col{j}{col.tolist()} = {dot.item():.2f}")
print()
print("out (1x4):", out[0].tolist(), "  <- same answer, two viewpoints")


# ============================================================
# Now a 2x3 weights matrix (TWO query tokens) @ the same V (3x4)
# ============================================================
print("\n" + "=" * 55)
print("2x3 weights @ 3x4 V  ->  2x4 output (2 query tokens)")
print("=" * 55)

scores = torch.tensor([[3.0, 1.0, 0.0],     # query token A's raw scores
                       [0.0, 0.0, 5.0]])    # query token B's raw scores
weights2 = scores.softmax(dim=-1)           # 2x3, each ROW sums to 1

out2 = weights2 @ V                          # (2x3) @ (3x4) -> 2x4

print("weights2 (2x3), each row sums to 1:")
for i, r in enumerate(weights2.tolist()):
    print(f"   query{i}: {[round(x,2) for x in r]}")
print("\nout2 (2x4) — each query gets its OWN blend of V's rows:")
for i, r in enumerate(out2.tolist()):
    print(f"   query{i}: {[round(x,2) for x in r]}")

print("\nVIEW 1 — inside computation (weighted sum of V's rows) per query:")
for i in range(weights2.shape[0]):
    w = weights2[i]
    print(f"\n  query{i}:  out = {w[0]:.2f}*row0 + {w[1]:.2f}*row1 + {w[2]:.2f}*row2")
    print(f"     {w[0]:.2f} * {V[0].tolist()} = {[round(x,2) for x in (w[0]*V[0]).tolist()]}")
    print(f"   + {w[1]:.2f} * {V[1].tolist()} = {[round(x,2) for x in (w[1]*V[1]).tolist()]}")
    print(f"   + {w[2]:.2f} * {V[2].tolist()} = {[round(x,2) for x in (w[2]*V[2]).tolist()]}")
    print(f"   = {[round(x,2) for x in out2[i].tolist()]}")
