"""demo_one_node.py — ONE node, peeled like an onion.

The node: PowBackward0, from loss = u^2  (u = w*x - 10 = -4, so loss = 16).

  SHELL 1 (outer):  the node's BEHAVIOR      — call it, a gradient comes out
  SHELL 2 (middle): the node's FORMULA       — the derivative rule + its saved ingredients
  SHELL 3 (core):   where the formula COMES FROM — the wiggle test: slope measured raw,
                    no calculus, proving the formula tells the truth
"""
import torch

# build the tiniest graph that ends in ^2
w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)
u = w * x - 10.0                 # u = -4
loss = u ** 2                    # loss = 16
node = loss.grad_fn              # PowBackward0 — today's onion

print("the op under study:  loss = u^2   with u =", u.item())
print("the node           :", type(node).__name__)

# ================================================================= SHELL 1
print("\n" + "=" * 64)
print("SHELL 1 — BEHAVIOR: gradient in -> gradient out. Just call it.")
print("=" * 64)
grad_out = node(torch.tensor(1.0))
print(f"   node(1.0)  ->  {grad_out.item()}")
print("   that's the whole outer shell: seed 1.0 in, blame -8 out.")
print("   but WHERE did -8 come from? peel.")

# ================================================================= SHELL 2
print("\n" + "=" * 64)
print("SHELL 2 — THE FORMULA inside, and its saved ingredients")
print("=" * 64)
print("   calculus rule for powers:   d(u^n)/du = n * u^(n-1)")
print("\n   the node SAVED its ingredients during forward:")
print(f"     _saved_self     = {node._saved_self.item()}    <- u, the input it squared")
print(f"     _saved_exponent = {node._saved_exponent}     <- n")
n_ = node._saved_exponent
u_ = node._saved_self.item()
by_formula = 1.0 * (n_ * u_ ** (n_ - 1))
print(f"\n   evaluate:  grad_in * n * u^(n-1)  =  1.0 * {n_} * ({u_})^{n_ - 1}  =  {by_formula}")
print(f"   matches SHELL 1's output? {by_formula == grad_out.item()}")
print("   -> the node is nothing but this one-line formula plus its saved numbers.")
print("      but WHY is the formula 2u? peel again.")

# ================================================================= SHELL 3
print("\n" + "=" * 64)
print("SHELL 3 — THE CORE: the wiggle test (no calculus allowed)")
print("=" * 64)
print("   a derivative CLAIMS: 'wiggle u by tiny h, output moves by (2u)*h'")
print("   don't trust it — measure it:\n")
def f(v):
    return v ** 2
print(f"   {'h':>10} {'(f(u+h)-f(u)) / h':>22}")
for h in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    slope = (f(u_ + h) - f(u_)) / h
    print(f"   {h:>10} {slope:>22.4f}")
print(f"\n   -> as the wiggle shrinks, the measured slope walks straight to {2 * u_}")
print("      = 2u. THAT is all the formula is: the answer this measurement")
print("      approaches, captured once as algebra so nobody ever has to")
print("      re-measure. calculus = pre-computed wiggle tests.")

# ================================================================= closing
print("\n" + "=" * 64)
print("the onion, reassembled")
print("=" * 64)
print("""   SHELL 1: node(grad_in) -> grad_out                 (behavior)
   SHELL 2: grad_out = grad_in * n*u^(n-1), u,n saved  (formula + homework)
   SHELL 3: the formula = the limit of measured wiggles (ground truth)

   every one of your model's 104 nodes is this same onion with a
   different SHELL 2 formula: Mul saves x and uses d(wx)/dw = x,
   Softmax saves its output, Embedding routes by saved indices...
   backward() = call 104 SHELL 1s, chained. Nothing else.""")
