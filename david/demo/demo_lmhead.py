"""demo_lmhead.py — poke at the lm_head: nn.Linear(hidden_size, vocab_size, bias=False)."""
import torch

hidden_size, vocab_size = 8, 6        # tiny so it's readable (real GLM: 6144, 154880)

lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

print("the module:", lm_head)
print("weight shape:", tuple(lm_head.weight.shape), " -> (vocab, hidden): one ROW per word")
print("bias:", lm_head.bias, " (None, because bias=False)")
print("total params:", lm_head.weight.numel(), f"= {vocab_size} x {hidden_size}\n")

print("the weight matrix (random until trained):")
print(lm_head.weight)

print("\nrow 2 = the 'output vector' for vocab word #2:")
print(lm_head.weight[2])

# how it's USED: hidden vector -> one logit per word
h = torch.tensor([1., 0., 2., -1., 0., 1., 0., 1.])    # one token's hidden vector (8 dims)
logits = lm_head(h)                                     # = h @ weight.T  -> 6 logits
print("\none hidden vector (8) -> logits (6):", [round(v, 2) for v in logits.tolist()])
print("each logit = h . that word's row  (dot product = how well they match)")
print("manual check, logit for word 2 =", round(float(h @ lm_head.weight[2]), 2))
