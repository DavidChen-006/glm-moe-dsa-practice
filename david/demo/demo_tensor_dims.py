"""demo_tensor_dims.py — learn what dim=0 / 1 / -1 mean until you can read any call.

Rule of thumb: dim = "the axis I'm operating ALONG; every other axis is kept as batch."

Run:  python david/demo/demo_tensor_dims.py
"""
import torch

# =============================================================================
# 0. Read a shape like a pro
# =============================================================================
print("=" * 60)
print("0. SHAPE = ordered list of axis sizes")
print("=" * 60)

# Give axes NAMES in your head whenever you see a shape.
# Example MoE scores: (T, E) = tokens × experts
x = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],   # token 0
        [10.0, 20.0, 30.0, 40.0],  # token 1
    ]
)  # shape (2, 4) → dim0=token, dim1=expert

print("x shape:", tuple(x.shape), "  ← (tokens=2, experts=4)")
print(x)
print("""
axis index:     dim=0          dim=1  (also dim=-1)
axis name:      tokens         experts
size:           2              4

Negative dims count from the END:
  dim=-1 → last axis     (experts here)
  dim=-2 → second-to-last (tokens here)
  For a 2D tensor: dim=-1 == dim=1,  dim=-2 == dim=0
""")


# =============================================================================
# 1. sum(dim=...) — collapses ONE axis
# =============================================================================
print("=" * 60)
print("1. sum(dim=...) collapses the axis you name")
print("=" * 60)

print("sum(dim=0)  → sum ACROSS tokens, keep experts")
print("  each column: token0 + token1")
s0 = x.sum(dim=0)
print(" ", tuple(s0.shape), s0.tolist(), "  ← one number per expert\n")

print("sum(dim=1)  → sum ACROSS experts, keep tokens")
print("  each row: expert0+1+2+3")
s1 = x.sum(dim=1)
print(" ", tuple(s1.shape), s1.tolist(), "  ← one number per token\n")

print("sum(dim=-1) → same as sum(dim=1) on 2D (last axis)")
assert torch.equal(x.sum(dim=-1), x.sum(dim=1))
print("  PASS: dim=-1 == dim=1 for this 2D tensor\n")


# =============================================================================
# 2. softmax(dim=...) — normalize ALONG an axis so that axis sums to 1
# =============================================================================
print("=" * 60)
print("2. softmax(dim=...) — 'make a distribution along this axis'")
print("=" * 60)

scores = torch.tensor(
    [
        [1.0, 2.0, 3.0],   # token 0's scores over 3 experts
        [3.0, 1.0, 1.0],   # token 1
    ]
)

print("scores", tuple(scores.shape))
print(scores)

sm_last = scores.softmax(dim=-1)
print("\nsoftmax(dim=-1) — each ROW becomes a distribution over experts:")
print(sm_last)
print("row sums:", sm_last.sum(dim=-1).tolist(), "  ← should be ~[1, 1]")

sm0 = scores.softmax(dim=0)
print("\nsoftmax(dim=0) — each COLUMN becomes a distribution over tokens:")
print(sm0)
print("col sums:", sm0.sum(dim=0).tolist(), "  ← should be ~[1, 1, 1]")
print("""
MoE / attention almost always want dim=-1:
  'for each token (row), turn expert/key scores into probabilities.'
""")


# =============================================================================
# 3. topk(dim=...) — pick winners ALONG an axis
# =============================================================================
print("=" * 60)
print("3. topk(k, dim=...) — rank along that axis only")
print("=" * 60)

scores = torch.tensor(
    [
        [0.1, 0.9, 0.3, 0.2],  # token 0
        [0.8, 0.1, 0.7, 0.0],  # token 1
    ]
)
print("scores", tuple(scores.shape))
print(scores)

vals, idxs = scores.topk(k=2, dim=-1)
print("\ntopk(2, dim=-1) — for EACH token, best 2 experts:")
print("  values ", vals.tolist())
print("  indices", idxs.tolist(), "  ← expert ids")

vals0, idxs0 = scores.topk(k=1, dim=0)
print("\ntopk(1, dim=0) — for EACH expert, best 1 token:")
print("  values ", vals0.tolist())
print("  indices", idxs0.tolist(), "  ← token ids")
print("""
Same numbers, different question:
  dim=-1 → 'which experts win for this token?'   (router)
  dim=0  → 'which token likes this expert most?' (different problem)
""")


# =============================================================================
# 4. 3D: dim=0 / 1 / 2 / -1
# =============================================================================
print("=" * 60)
print("4. THREE dimensions — MoE group reshape playground")
print("=" * 60)

# (T=2 tokens, G=2 groups, Eg=3 experts-per-group)  → E = 6 total experts
g = torch.arange(12, dtype=torch.float).view(2, 2, 3)
print("g shape (T, G, Eg) =", tuple(g.shape))
print(g)
print("""
dim index:   0       1       2   (== -1)
name:        token   group   expert-in-group
""")

print("sum(dim=-1) → sum experts inside each group → shape (T, G)")
print(" ", tuple(g.sum(dim=-1).shape), g.sum(dim=-1).tolist())

print("sum(dim=1)  → sum across groups → shape (T, Eg)")
print(" ", tuple(g.sum(dim=1).shape), g.sum(dim=1).tolist())

print("sum(dim=0)  → sum across tokens → shape (G, Eg)")
print(" ", tuple(g.sum(dim=0).shape), g.sum(dim=0).tolist())

print("\ntopk(2, dim=-1)[0] — top-2 expert scores *inside each group*")
top2 = g.topk(2, dim=-1)[0]
print("  shape", tuple(top2.shape), "← (T, G, 2)")
print(top2)
print("  .sum(dim=-1) → group_scores shape", tuple(top2.sum(dim=-1).shape))
print(" ", top2.sum(dim=-1).tolist())
print("  ↑ this is exactly the MoE group_scores idea\n")


# =============================================================================
# 5. view / reshape — new axis layout, same elements
# =============================================================================
print("=" * 60)
print("5. view/reshape — rename how elements are grouped (no new data)")
print("=" * 60)

flat = torch.arange(12, dtype=torch.float)  # (12,)
print("flat", tuple(flat.shape), flat.tolist())

as_TE = flat.view(2, 6)                     # (T=2, E=6)
print("view(2, 6)     → (T, E)     ", tuple(as_TE.shape))
print(as_TE)

as_TGE = as_TE.view(2, 2, 3)                # (T, G, Eg)
print("view(2, 2, 3)  → (T, G, Eg) ", tuple(as_TGE.shape))
print(as_TGE)

print("""
Reading .view(-1, n_group, E // n_group) on (T, E):
  -1        → 'keep whatever T is' (infer from numel)
  n_group   → new middle axis G
  E//n_group→ experts per group
Same  T*E  numbers, now addressed as (token, group, expert-in-group).
""")


# =============================================================================
# 6. Quick drills — predict, then check
# =============================================================================
print("=" * 60)
print("6. DRILLS — fill the mental blank, then assert")
print("=" * 60)

a = torch.arange(6.0).view(2, 3)
# [[0,1,2],
#  [3,4,5]]

assert a.sum(dim=0).tolist() == [3.0, 5.0, 7.0], "sum dim0 = column sums"
assert a.sum(dim=1).tolist() == [3.0, 12.0], "sum dim1 = row sums"
assert a.sum(dim=-1).tolist() == [3.0, 12.0], "dim=-1 is last axis"

b = torch.tensor([[1.0, 100.0, 2.0], [50.0, 3.0, 4.0]])
_, idx = b.topk(1, dim=-1)
assert idx.tolist() == [[1], [0]], "dim=-1 top1 = best expert per token"

_, idx0 = b.topk(1, dim=0)
assert idx0.tolist() == [[1, 0, 1]], "dim=0 top1 = best token per expert"

c = torch.arange(8.0).view(2, 4)
assert tuple(c.view(2, 2, 2).shape) == (2, 2, 2)
assert torch.equal(c.view(2, 2, 2).reshape(2, 4), c)

print("PASS: all dim drills")
print("""
CHEAT SHEET
-----------
dim=0     first axis
dim=1     second axis
dim=-1    last axis   ← default for 'per row' ops (softmax/topk in NLP)
dim=-2    second-to-last

When you see  foo(x, dim=D):
  1. Name the axes of x's shape.
  2. D is the axis that gets reduced / ranked / normalized.
  3. All other axes stay (batch structure).
""")
