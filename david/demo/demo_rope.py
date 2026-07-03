"""demo_rope.py — what goes IN and OUT of RotaryEmbedding (RoPE), stage by stage.

Two pieces, matching the reference modeling file:
  1. RotaryEmbedding (the class):  position ids IN  ->  (cos, sin) tables OUT. No rotating.
  2. apply_rotary_pos_emb (the fn): Q, K, cos, sin IN -> rotated Q, K OUT. Same shapes.

Then the payoff: after rotation, Q.K dot products depend on RELATIVE distance.
"""
import torch

torch.set_printoptions(precision=3, sci_mode=False)

# toy sizes: 1 batch, 6 token positions, head_dim=8 (so 4 rotating pairs = 4 "clock hands")
SEQ, HEAD_DIM, THETA = 6, 8, 10000.0

# ---------- STAGE 1: RotaryEmbedding — positions in, (cos, sin) out ----------
# inv_freq = the spin SPEED of each pair: fast hands first, slow hands last
inv_freq = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))   # (4,)
print("INPUT  inv_freq (spin speed per pair, fast -> slow):\n ", inv_freq)

position_ids = torch.arange(SEQ)                        # (6,)  <- the INPUT: just 0,1,2,3,4,5
print("\nINPUT  position_ids:", position_ids.tolist())

angles = torch.outer(position_ids.float(), inv_freq)    # (6, 4): angle = position * speed
cos = torch.cat([angles.cos(), angles.cos()], dim=-1)   # (6, 8)
sin = torch.cat([angles.sin(), angles.sin()], dim=-1)   # (6, 8)
print("\nOUTPUT cos table, shape", tuple(cos.shape), "(one row per POSITION, no token content!):\n", cos)
print("\n -> the class is an ANGLE FACTORY: it never sees Q, K, or hidden states.")

# ---------- STAGE 2: apply_rotary_pos_emb — Q, K in, rotated Q, K out ----------
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

torch.manual_seed(0)
q = torch.randn(SEQ, HEAD_DIM)    # one query vector per position
k = q.clone()                     # SAME content at every step -> isolates the position effect

q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
print("\nINPUT  q shape:", tuple(q.shape), " OUTPUT q_rot shape:", tuple(q_rot.shape), " <- shape unchanged")

# rotation never changes length, only direction:
print("norm per position before:", [round(v, 3) for v in q.norm(dim=-1).tolist()])
print("norm per position after: ", [round(v, 3) for v in q_rot.norm(dim=-1).tolist()])

# ---------- PAYOFF: dot products now encode RELATIVE distance ----------
# make every position hold the IDENTICAL vector, so any score difference is purely positional
v = torch.randn(HEAD_DIM)
same = v.expand(SEQ, HEAD_DIM)
sq, sk = apply_rotary_pos_emb(same, same, cos, sin)
scores = sq @ sk.T                                      # (6, 6) all-pairs dot products

print("\nScores between IDENTICAL vectors at different positions (before RoPE all would be",
      f"{(v @ v).item():.3f}):\n", scores)
print("\ndiagonal    (distance 0):", [round(scores[i, i].item(), 3) for i in range(SEQ)])
print("one-off     (distance 1):", [round(scores[i, i + 1].item(), 3) for i in range(SEQ - 1)])
print("two-off     (distance 2):", [round(scores[i, i + 2].item(), 3) for i in range(SEQ - 2)])
print("\n -> constant along each diagonal: the score depends ONLY on how far apart two",
      "\n    tokens are, not on WHERE they sit. That's the whole point of RoPE.")
