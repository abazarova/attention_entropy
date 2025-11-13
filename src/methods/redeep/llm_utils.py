import torch
import torch.nn.functional as F
import gc
from functools import wraps
from typing import Optional, Tuple


class WrapperForResidualStreamLogits:
    def __init__(self):
        self.logits = []
        self.model = None
        self.num_hidden_layers = None

    def reset(self):
        del self.logits
        torch.cuda.empty_cache()
        gc.collect()
        self.logits = []

    def post_init(self, model: "AutoModelForCasualLM"):
        self.model = model
        
    def _attend_residual_stream(self, forward):
        @wraps(forward)
        def wrapper(*args, **kwargs):
            output, (anchor_hiddens, mature_hiddens) = forward(*args, **kwargs)

            anchor_hiddens = self._norm(anchor_hiddens)
            mature_hiddens = self._norm(mature_hiddens)

            anchor_logits = self._compute_logits(anchor_hiddens)
            mature_logits = self._compute_logits(mature_hiddens)

            anchor_logits = anchor_logits.detach().cpu()
            mature_logits = mature_logits.detach().cpu()

            self.logits.append((anchor_logits, mature_logits))
            return output
        return wrapper
    
    def _compute_logits(self, hidden_states, logits_to_keep=0):
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model.lm_head(hidden_states[:, slice_indices, :])
        return logits
        
    def _norm(self, hiddens):
        return self.model.model.norm(hiddens)
    
    def __iter__(self):
        return iter(self.logits)


wrapper = WrapperForResidualStreamLogits()

#compatible with exactly transformers==4.43
@wrapper._attend_residual_stream
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    anchor_hiddens = hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    mature_hiddens = hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs, (anchor_hiddens, mature_hiddens)