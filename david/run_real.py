import torch
from modeling_glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM   # YOUR classes

config = GlmMoeDsaConfig(vocab_size=96, hidden_size=128, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=256)   # toy sizes
                         
model = GlmMoeDsaForCausalLM(config)                  # build instance (PyTorch allocates the weights)

input_ids = torch.randint(0, config.vocab_size, (1, 5))   # fake prompt: 1 batch, 5 tokens
logits = model(input_ids)                              # run forward → PyTorch does all the matmuls
                                                       # logits shape: (1, 5, vocab)

next_token = logits[0, -1].argmax()                    # last token's prediction
print("predicted token id:", next_token.item())