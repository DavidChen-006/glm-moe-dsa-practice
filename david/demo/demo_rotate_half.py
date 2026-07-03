"""demo_rotate_half.py — how ONE elementwise line performs the classic 2-line rotation
formula on every pair at once.

The claim: x*cos + rotate_half(x)*sin  ==  {new_x = x·cos − y·sin ; new_y = x·sin + y·cos}
applied to all pairs simultaneously. We prove it slot by slot.
"""
import math

import torch

torch.set_printoptions(precision=4, sci_mode=False)

# ============ SETUP: one token row, 4 dims -> 2 pairs (half-split layout) ============
row = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("token row:", row.tolist(), "  (4 dims)")
print("""
PAIRING (GLM's half-split convention: dim i partners with dim i + half):
    pair A = (slot0, slot2) = (1.0, 3.0)   <- one little (x, y) point
    pair B = (slot1, slot3) = (2.0, 4.0)   <- another (x, y) point
""")

# one angle per pair (in the real model: angle = position * that pair's speed)
angA, angB = math.radians(30), math.radians(30)   # same angle for both pairs, to keep it easy
cos = torch.tensor([math.cos(angA), math.cos(angB), math.cos(angA), math.cos(angB)])
sin = torch.tensor([math.sin(angA), math.sin(angB), math.sin(angA), math.sin(angB)])
print("angle = 30 degrees for every pair (real model: each pair has its own)")
print("cos table:", [round(c, 4) for c in cos.tolist()], " <- note slots 0&2 share, 1&3 share")
print("sin table:", [round(s, 4) for s in sin.tolist()])

# ============ STEP 1: the internet's classic formula, pair by pair, by hand ============
print("\n" + "=" * 72)
print("STEP 1 — the classic 2-line formula, applied to each pair BY HAND")
print("=" * 72)
print("   new_x = x*cos - y*sin")
print("   new_y = x*sin + y*cos      <- BOTH outputs; rotation always makes two\n")
c, s = math.cos(angA), math.sin(angA)
for name, x, y in [("A", 1.0, 3.0), ("B", 2.0, 4.0)]:
    nx = x * c - y * s
    ny = x * s + y * c
    print(f"  pair {name}: (x={x}, y={y})")
    print(f"     new_x = {x}*{c:.4f} - {y}*{s:.4f} = {nx:.4f}")
    print(f"     new_y = {x}*{s:.4f} + {y}*{c:.4f} = {ny:.4f}")
by_hand = torch.tensor([1.0 * c - 3.0 * s, 2.0 * c - 4.0 * s, 1.0 * s + 3.0 * c, 2.0 * s + 4.0 * c])
print("\n  by-hand result, placed back into slots [newA_x, newB_x, newA_y, newB_y]:")
print("  ", [round(v, 4) for v in by_hand.tolist()])

# ============ STEP 2: rotate_half — deliver each slot its PARTNER, signed ============
print("\n" + "=" * 72)
print("STEP 2 — rotate_half(x): swap halves, negate the incoming half")
print("=" * 72)
def rotate_half(x):
    x1, x2 = x[: x.shape[-1] // 2], x[x.shape[-1] // 2 :]
    return torch.cat((-x2, x1))

rh = rotate_half(row)
print(f"""
   input  slots: [ 1,  2,  3,  4 ]
   output slots: {rh.tolist()}

   slot 0 now holds -3  <- pair A's partner (y), negated
   slot 1 now holds -4  <- pair B's partner (y), negated
   slot 2 now holds  1  <- pair A's partner (x), as-is
   slot 3 now holds  2  <- pair B's partner (x), as-is

   That's ALL it does: line up each slot's partner with the sign the formula needs.
   (-y goes where new_x is computed; +x goes where new_y is computed)""")

# ============ STEP 3: the one-liner, term by term ============
print("=" * 72)
print("STEP 3 — the tensor one-liner:  x*cos + rotate_half(x)*sin,  term by term")
print("=" * 72)
term1 = row * cos
term2 = rh * sin
one_liner = term1 + term2
print(f"""
   term1 = x * cos             = {[round(v, 4) for v in term1.tolist()]}
           slot0: 1*cos (=x*cos for A)      slot2: 3*cos (=y*cos for A)

   term2 = rotate_half(x)*sin  = {[round(v, 4) for v in term2.tolist()]}
           slot0: -3*sin (=-y*sin for A)    slot2: 1*sin (=x*sin for A)

   sum   = term1 + term2       = {[round(v, 4) for v in one_liner.tolist()]}
           slot0: x*cos - y*sin  = new_x  of pair A
           slot2: x*sin + y*cos  = new_y  of pair A   (same story for B in slots 1,3)""")

# ============ THE VERDICT ============
print("=" * 72)
print("VERDICT — do the two methods agree, slot for slot?")
print("=" * 72)
print(f"\n   classic formula by hand: {[round(v, 4) for v in by_hand.tolist()]}")
print(f"   tensor one-liner:        {[round(v, 4) for v in one_liner.tolist()]}")
print(f"   identical? {torch.allclose(by_hand, one_liner)}")
print("""
   One elementwise pass = the classic 2-line rotation formula executed on
   every pair at once. rotate_half is just the partner-delivery service.
   In the real model: 64 pairs instead of 2, each with its own angle
   (position x that pair's speed) -- same three steps exactly.""")
