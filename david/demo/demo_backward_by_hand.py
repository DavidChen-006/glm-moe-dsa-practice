"""demo_backward_by_hand.py — LAYER 2: execute backprop yourself, one node at a time.

Layer 1 taught: a grad_fn node is callable — upstream gradient IN, local gradient OUT.
Here we build a tiny graph, then walk it BY HAND, calling each node ourselves and
checking every number against pencil-and-paper calculus. No loss.backward() until
the very end — and only to prove our hand-walk got the same answer.

The tiny model:   y = w * x        (w=3, x=2  ->  y=6)
                  loss = (y - 10)^2            ->  loss=16
Calculus says:    dloss/dy = 2*(y-10) = -8
                  dloss/dw = dloss/dy * x = -8 * 2 = -16
"""
import torch

w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)
y = w * x
loss = (y - 10.0) ** 2

print("the tape, tip to mailbox:")
print("   loss <- PowBackward0 <- SubBackward0 <- MulBackward0 <- AccumulateGrad(w)")
print(f"   loss = {loss.item()}   w.grad = {w.grad}   <- mailbox empty, nothing run yet")

# ---------------------------------------------------------------- step 1
print("\n" + "=" * 66)
print("STEP 1 — call PowBackward0 by hand (the ^2 node)")
print("=" * 66)
pow_node = loss.grad_fn
print("node:", type(pow_node).__name__)
grad_in = torch.tensor(1.0)          # dloss/dloss = 1: the walk always starts with 1
grad_out = pow_node(grad_in)
print(f"   IN : {grad_in.item()}   (dloss/dloss — the seed)")
print(f"   OUT: {grad_out.item()}   = dloss/d(y-10)")
print(f"   calculus check: d(u^2)/du = 2u = 2*(-4) = -8   {'OK' if grad_out.item() == -8 else 'MISMATCH'}")

# ---------------------------------------------------------------- step 2
print("\n" + "=" * 66)
print("STEP 2 — follow next_functions, call SubBackward0 (the y-10 node)")
print("=" * 66)
sub_node = pow_node.next_functions[0][0]
print("node:", type(sub_node).__name__, "  <- found via pow_node.next_functions")
outs = sub_node(grad_out)            # feed step 1's output in — THE CHAIN
print("   raw output:", outs, "  <- a TUPLE: one gradient per input of the op!")
print("   (y-10 had two inputs: y and the constant 10; the constant needs no gradient)")
grad_out2 = outs[0]
print(f"   IN : {grad_out.item()}")
print(f"   OUT: {grad_out2.item()}   = dloss/dy")
print(f"   calculus check: d(y-10)/dy = 1, so grad passes through unchanged: -8   "
      f"{'OK' if grad_out2.item() == -8 else 'MISMATCH'}")

# ---------------------------------------------------------------- step 3
print("\n" + "=" * 66)
print("STEP 3 — follow again, call MulBackward0 (the w*x node)")
print("=" * 66)
mul_node = sub_node.next_functions[0][0]
print("node:", type(mul_node).__name__)
outs = mul_node(grad_out2)           # again a tuple: gradients for (w, x)
print("   raw output:", outs, "  <- gradient for w, and None for x (x doesn't require grad)")
grad_w = outs[0]
print(f"   IN : {grad_out2.item()}")
print(f"   OUT: {grad_w.item()}   = dloss/dw")
print(f"   calculus check: d(w*x)/dw = x = 2, chain: -8 * 2 = -16   "
      f"{'OK' if grad_w.item() == -16 else 'MISMATCH'}")
print("   (note: this node SAVED x=2 during forward — its _saved homework — ")
print("    that is exactly why ops stash their inputs)")

# ---------------------------------------------------------------- step 4
print("\n" + "=" * 66)
print("STEP 4 — the mailbox: deliver by calling AccumulateGrad")
print("=" * 66)
acc_node = mul_node.next_functions[0][0]
print("node:", type(acc_node).__name__, "  <- the leaf, w's mailbox")
print(f"   w.grad BEFORE: {w.grad}")
acc_node(grad_w)                     # deposit the blame
print(f"   w.grad AFTER : {w.grad.item()}   <- WE just did what backward() does")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 66)
print("VERDICT — rebuild the same graph, let autograd do it, compare")
print("=" * 66)
w2 = torch.tensor(3.0, requires_grad=True)
loss2 = (w2 * x - 10.0) ** 2
loss2.backward()                     # the machine
print(f"   our hand-walk delivered : {w.grad.item()}")
print(f"   loss.backward() delivers: {w2.grad.item()}")
print(f"   identical? {w.grad.item() == w2.grad.item()}")
print("""
 -> backward() is EXACTLY the loop you just did by hand:
    seed 1.0 at the tip -> call node -> follow next_functions -> call parent
    -> repeat -> deposit at AccumulateGrad mailboxes.
    On your GLM model it is the same walk, just 104 nodes and 11 mailboxes.""")
