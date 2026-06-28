from collections.abc import Callable

from .glm_moe_dsa_pretrained_model import GlmMoeDsaPreTrainedModel
from transformers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class GlmMoeDsaRMSNorm(nn.Module):


class GlmMoeDsaIndexer(nn.Module):


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
    
    forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, #attn_output softmax(QKt/d + M) * V
               torch.Tensor, #attn_weights softmax(QKt/d + M)
               torch.Tensor, 
                | None]:
            
        query_states =  hidden_states @ Wq
        key_states =  hidden_states @ Wk
        value_states = hidden_states @ Wv

        combined_mask = 

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_inteface( self.config._attn_implementaion, eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            scaling=self.scaling,
            **kwargs,
        )

        return attn_output, attn_weights


class GlmMoeDsaMLP(nn.Module):


class GlmMoeDsaTopkRouter(nn.Module):


class GlmMoeDsaNaiveMoe(nn.Module):


class GlmMoeDsaMoE(nn.Module):


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

        self.mlp = GlmMoeDsaMLP()

        self.input_layernorm = GlmMoeDsaRMSNorm()

        self.post_attention_layernorm = GlmMoeDsaRMSNorm()

    forward() -> :



class GlmMoeDsaRotaryEmbedding(nn.Module):


class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmMoeDsaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False



class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
