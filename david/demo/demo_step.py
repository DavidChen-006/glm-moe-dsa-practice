"""demo_step.py — inspect optimizer.step() on a small weight matrix, cell by cell.

Given a matrix of weights and their gradients (from backprop), watch step() transform
each weight independently. Three parts:
  PART 1: the equation by hand, per cell, with direction annotations
  PART 2: verify the real torch SGD.step() produces the identical result
  PART 3: the honest twist — AdamW (what train.py actually uses) does something MORE
"""
import torch

torch.set_printoptions(precision=4, sci_mode=False)

lr = 0.1

# a 2x3 weight matrix, and its gradients (from backprop) — hand-picked to be legible:
# a big negative, a small positive, a zero, etc.
W0 = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
G = torch.tensor([[-10.0, 0.5, 0.0],
                  [2.0, -8.0, 4.0]])

print("WEIGHTS (W):")
print(W0)
print("\nGRADIENTS (W.grad) — one per weight, from backprop:")
print(G)
print(f"\nlearning rate = {lr}   (ONE shared scalar for every weight)")

# ================================================================ PART 1
print("\n" + "=" * 74)
print("PART 1 — the equation per cell:   w_new = w - lr * grad")
print("=" * 74)
print(f"{'cell':>6} {'w':>7} {'grad':>8} {'direction':>26} {'change':>9} {'w_new':>8}")
for i in range(2):
    for j in range(3):
        w, g = W0[i, j].item(), G[i, j].item()
        change = -lr * g
        if g > 0:
            d = "grad>0 (loss↑ if w↑) -> w DOWN"
        elif g < 0:
            d = "grad<0 (loss↑ if w↓) -> w UP"
        else:
            d = "grad=0 -> frozen"
        print(f"  [{i}][{j}] {w:>7.2f} {g:>8.2f} {d:>30} {change:>+9.3f} {w - lr*g:>8.3f}")
print("\n  same lr for all six, yet each moved differently — because the GRADIENT differed.")
print("  the minus sign flipped each gradient's own sign: + grad -> down, - grad -> up.")

# ================================================================ PART 2
print("\n" + "=" * 74)
print("PART 2 — verify: the real torch SGD.step() does exactly this")
print("=" * 74)
W = W0.clone().requires_grad_(True)
W.grad = G.clone()
by_hand = (W0 - lr * G)
opt = torch.optim.SGD([W], lr=lr)
opt.step()                                  # <- the real thing, in place
print("by hand (W - lr*grad):")
print(by_hand)
print("\nafter SGD.step():")
print(W.data)
print("\nidentical?", torch.allclose(by_hand, W.data))
print(" -> optimizer.step() with plain SGD IS the equation, run over every weight.")

# ================================================================ PART 3
print("\n" + "=" * 74)
print("PART 3 — the twist: AdamW (what train.py uses) does something MORE")
print("=" * 74)
W2 = W0.clone().requires_grad_(True)
W2.grad = G.clone()
optA = torch.optim.AdamW([W2], lr=lr, weight_decay=0.0)   # decay off, to isolate the effect
optA.step()
sgd_change = (W.data - W0)
adam_change = (W2.data - W0)
print("SGD change per weight  (= -lr*grad, so PROPORTIONAL to gradient size):")
print(sgd_change)
print("\nAdamW change per weight (first step):")
print(adam_change)
print(f"""
  Look at the sizes. SGD moved the grad=-10 weight by {sgd_change[0,0].item():+.3f} and the
  grad=0.5 weight by only {sgd_change[0,1].item():+.3f} — proportional to the gradient.

  AdamW moved BOTH by about the same ~{abs(adam_change[0,0].item()):.2f} (~lr), regardless of
  gradient magnitude — it normalizes each weight by its own gradient size. Same
  DIRECTION as SGD (sign preserved), but the step size is rescaled per-weight.

  So: the equation w -= lr*grad is the SKELETON. Plain SGD is it, naked. AdamW keeps the
  skeleton (subtract a step, direction from the gradient) but replaces the raw gradient
  with a momentum-smoothed, per-weight-rescaled version. (You wrote AdamW's real math out
  by hand in minimind/train_pretrain_fp.py's adamw_update.)""")

# ---- closing: the lifecycle ----
print("\n" + "=" * 74)
print("and after the step: zero_grad wipes the mailboxes for the next batch")
print("=" * 74)
print("W.grad before zero_grad:")
print(W.grad)
opt.zero_grad()
print("W.grad after  zero_grad:", W.grad)
print(" -> next batch computes fresh gradients here; this step's blame is spent.")
