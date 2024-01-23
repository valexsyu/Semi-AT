import torch
from transformers import LlamaForCausalLM, LlamaModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
    is_torch_fx_available,
)
from transformers.cache_utils import Cache,DynamicCache
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from utils.train_speedup_utils import (
    _prepare_4d_causal_attention_mask_semi_at, 
    _prepare_4d_causal_attention_mask_for_sdpa_semi_at,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from generation.utils import GenerationSemiAT



logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx
    _prepare_4d_causal_attention_mask_semi_at = torch.fx.wrap(_prepare_4d_causal_attention_mask_semi_at)



SEMI_LLAMA_INPUTS_DOCSTRING = r"""
    Use the config to get semi_at_attention. semi_at_attention is using special causal attention for training.
    Use semi_at_insert_token_id the get insert token id
     
    Args:
        the same with LLAMA.
"""


class LlamaSemiATModel(LlamaModel):
    def __init__(self, config, semi_at_insert_token_id:Optional[int] = None):
        super().__init__(config)    
        self.semi_at_attention = getattr(self.config, "semi_at_attention", False)
        self.semi_at_insert_token_id = semi_at_insert_token_id
        if self.semi_at_attention :
            self._prepare_4d_causal_attention_mask_for_sdpa_warp = _prepare_4d_causal_attention_mask_for_sdpa_semi_at
            self._prepare_4d_causal_attention_mask_warp = _prepare_4d_causal_attention_mask_semi_at
        else:
            self._prepare_4d_causal_attention_mask_for_sdpa_warp = _prepare_4d_causal_attention_mask_for_sdpa
            self._prepare_4d_causal_attention_mask_warp = _prepare_4d_causal_attention_mask
            



    @add_start_docstrings_to_model_forward(SEMI_LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = self._prepare_4d_causal_attention_mask_for_sdpa_warp(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        
        else:
            # 4d mask is passed through the layers
            attention_mask = self._prepare_4d_causal_attention_mask_warp(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )












        
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPast]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache

    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # retrieve input_ids and inputs_embeds
    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         batch_size, seq_length = input_ids.shape[:2]
    #     elif inputs_embeds is not None:
    #         batch_size, seq_length = inputs_embeds.shape[:2]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     past_key_values_length = 0
    #     if past_key_values is not None:
    #         past_key_values_length = past_key_values[0][0].shape[2]

    #     if position_ids is None:
    #         device = input_ids.device if input_ids is not None else inputs_embeds.device
    #         position_ids = torch.arange(
    #             past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    #         )
    #         position_ids = position_ids.unsqueeze(0)

    #     if inputs_embeds is None:
    #         inputs_embeds = self.embed_tokens(input_ids)

    #     if getattr(self.config, "_flash_attn_2_enabled", False):
    #         # 2d mask is passed through the layers
    #         # attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    #         raise ValueError("semi autoregressive is not support")
    #     else:
    #         insert_token_attention_mask = input_ids == self.semi_at_insert_token_id
    #         attention_mask = _prepare_4d_causal_attention_mask_semi_at(
    #             attention_mask, (batch_size, seq_length), 
    #             inputs_embeds, past_key_values_length ,
    #             self.semi_at_attention, insert_token_attention_mask,
    #         )            


    #     # embed positions
    #     hidden_states = inputs_embeds

    #     if self.gradient_checkpointing and self.training:
    #         if use_cache:
    #             logger.warning_once(
    #                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #             )
    #             use_cache = False

    #     # decoder layers
    #     all_hidden_states = () if output_hidden_states else None
    #     all_self_attns = () if output_attentions else None
    #     next_decoder_cache = () if use_cache else None

    #     for idx, decoder_layer in enumerate(self.layers):
    #         if output_hidden_states:
    #             all_hidden_states += (hidden_states,)

    #         past_key_value = past_key_values[idx] if past_key_values is not None else None

    #         if self.gradient_checkpointing and self.training:
    #             layer_outputs = self._gradient_checkpointing_func(
    #                 decoder_layer.__call__,
    #                 hidden_states,
    #                 attention_mask,
    #                 position_ids,
    #                 past_key_value,
    #                 output_attentions,
    #                 use_cache,
    #             )
    #         else:
    #             layer_outputs = decoder_layer(
    #                 hidden_states,
    #                 attention_mask=attention_mask,
    #                 position_ids=position_ids,
    #                 past_key_value=past_key_value,
    #                 output_attentions=output_attentions,
    #                 use_cache=use_cache,
    #             )

    #         hidden_states = layer_outputs[0]

    #         if use_cache:
    #             next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

    #         if output_attentions:
    #             all_self_attns += (layer_outputs[1],)

    #     hidden_states = self.norm(hidden_states)

    #     # add hidden states from the last decoder layer
    #     if output_hidden_states:
    #         all_hidden_states += (hidden_states,)

    #     next_cache = next_decoder_cache if use_cache else None
    #     if not return_dict:
    #         return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    #     return BaseModelOutputWithPast(
    #         last_hidden_state=hidden_states,
    #         past_key_values=next_cache,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attns,
    #     )    

class LlamaSemiATForCausalLM(LlamaForCausalLM, GenerationSemiAT):
    def __init__(self, config, semi_at_insert_token_id:Optional[int] = None):
        super().__init__(config)
        self.model=LlamaSemiATModel(config,semi_at_insert_token_id)
    
    
    
    
    
    def prepare_inputs_for_generation(
        self, input_ids, insert_token_num=0, previous_same_sequences_length=None,
        insert_token_id=None,
        past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            # reuse the cache only the previous sequences is not change for semi-AT 
            past_key_values = tuple([(layer[0][:, :, :previous_same_sequences_length, :],
                                      layer[1][:, :, :previous_same_sequences_length, :]) for layer in past_key_values]) 
            
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # add the insert token for smei at:
            # input_ids and attention_mask
                       
            
            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        if insert_token_num > 0 :
            ## add insert_token_id for semi-AT        
            input_ids = torch.cat([input_ids, torch.tensor(insert_token_id).to(input_ids.device).repeat(input_ids.size(0), insert_token_num)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.tensor(1).to(attention_mask.device).repeat(attention_mask.size(0), insert_token_num)], dim=1)                 

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    
