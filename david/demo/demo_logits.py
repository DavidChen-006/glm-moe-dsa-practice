"""demo_logits.py — the spine of how logits are made: hidden_states -> lm_head -> logits -> word."""
import torch
from torch.nn.functional import softmax

vocab = ["the", "cat", "dog", "sat", "ran", "."]   # tiny 6-word vocabulary
V, H = len(vocab), 4                                # vocab=6, hidden=4

# 3 tokens, each a 4-dim final hidden vector (pretend the model already produced these)
hidden_states = torch.tensor([
    [1.0, 0.0, 2.0, -1.0],   # token 0
    [0.0, 1.0, 0.0, 1.0],    # token 1
    [2.0, -1.0, 1.0, 0.0],   # token 2  (the LAST one -> next-word prediction)
])

lm_head = torch.nn.Linear(H, V, bias=False)        # hidden(4) -> vocab(6)

logits = lm_head(hidden_states)                     # (3 tokens, 6 vocab) -- a score per word, per token
print("hidden_states:", tuple(hidden_states.shape), " -> logits:", tuple(logits.shape), "(tokens x vocab)\n")

for t in range(3):
    probs = softmax(logits[t], dim=-1)
    pick = int(logits[t].argmax())
    print(f"token{t}: logits = {[round(v,2) for v in logits[t].tolist()]}")
    print(f"        probs  = {[round(v,2) for v in probs.tolist()]}  -> predicts '{vocab[pick]}'\n")

last = int(logits[-1].argmax())
print(f"NEXT-WORD PREDICTION (last token's logits): '{vocab[last]}'")
