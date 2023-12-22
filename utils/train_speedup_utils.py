import math
import torch
import random
from typing import List, Optional, Tuple, Union

from transformers.modeling_attn_mask_utils import AttentionMaskConverter

def inject_noise(batch, tokenizer, p):
    if p == 0.0:
        return batch

    # Assuming batch is a 2-dimensional tensor (batch_size x max_sequence_length)
    batch_size, max_sequence_length = batch.size()

    # Identify padding, BOS, and EOS indices
    padding_mask = (batch == tokenizer.pad_token_id)
    bos_mask = (batch == tokenizer.bos_token_id)
    eos_mask = (batch == tokenizer.eos_token_id)

    # Combine masks to get overall skip mask
    skip_mask = padding_mask | bos_mask | eos_mask

    # Calculate the number of non-padding, non-BOS, and non-EOS tokens per sequence
    num_non_skip_tokens_per_sequence = (max_sequence_length - skip_mask.sum(dim=1))
    # Calculate the number of tokens to inject noise into for each sequence
    num_tokens_to_inject = torch.ceil(num_non_skip_tokens_per_sequence * p).int()

    # Initialize result tensor with the original batch values
    result = batch.clone()

    for i in range(batch_size):
        tokens = batch[i, :]

        # Identify non-skip indices
        non_skip_indices = ~skip_mask[i, :]

        # Sample noise indices from non-skip indices
        noise_indices = torch.nonzero(non_skip_indices).squeeze().view(-1)
        noise_indices = noise_indices[torch.randperm(len(noise_indices))][:num_tokens_to_inject[i]]

        # Create noise mask
        noise_mask = torch.zeros_like(tokens, dtype=torch.bool)
        noise_mask[noise_indices] = 1

        # Sample random ratio
        random_ratio = random.uniform(0, 1)

        # Calculate the number of random tokens to inject
        num_random = torch.ceil(num_tokens_to_inject[i] * random_ratio).int()

        # Sample random indices for random tokens
        random_indices = torch.randint(low=3, high=len(tokenizer), size=(num_random,)).to(result.device)
        
        # Inject random tokens at selected positions
        result[i, noise_indices[:num_random]] = random_indices

        # Copy non-noise tokens from the original sequence
        result[i, non_skip_indices] = tokens[non_skip_indices]

        assert (result[i, :] >= 0).all()

    return result


def _prepare_4d_causal_attention_mask_semi_at(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    semi_at_attention: Optional[bool] = False,
    insert_token_attention_mask: Optional[bool] = None,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
        
    attn_mask_converter = SemiATAttentionMaskConverter(
                                is_causal=True, 
                                sliding_window=sliding_window,
                                semi_at_attention=semi_at_attention,
                                insert_token_attention_mask=insert_token_attention_mask
                                )

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask





class SemiATAttentionMaskConverter(AttentionMaskConverter):
    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None, 
                 semi_at_attention: Optional[bool] = False,
                 insert_token_attention_mask: Optional[bool] = None,
                 ):
        super().__init__(is_causal,sliding_window)
        if semi_at_attention :
            self._make_causal_mask = self._make_semi_at_causal_mask
            self.insert_token_attention_mask = insert_token_attention_mask
    
            
    
    def _make_semi_at_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        if past_key_values_length > 0:
            query_len = tgt_len + past_key_values_length
        else:
            query_len = tgt_len
        
        mask = torch.full((tgt_len, query_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)
            
        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, query_len)
        
        diagonal_bool =  torch.full((tgt_len, query_len), True, device=device)
        diagonal_bool.masked_fill_(mask_cond == mask_cond.view(tgt_len, 1), False)
        
        breakpoint()
        mask[(self.insert_token_attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, query_len) | mask ) &
             diagonal_bool[None, None, :, :].expand(bsz, 1, tgt_len, query_len)
                        ]        

        return mask   



def insert_semi_at_token(
    input_noise_ids: torch.LongTensor = None,
    input_ids: torch.LongTensor = None, 
    attention_mask: torch.Tensor = None, 
    insert_token_id: int= None,
    pad_id:int =None,
    insert_token_num : int = 0,
    position_ids: Optional[torch.LongTensor] = None,
    sequence_label_ingnore: bool = False,
    ):

    # start from second tokens and window sliding
    # ex: /s A B C D E p insert_token_num=1
    #               start----            -----end
    #                        |           |
    # input_ids     = /s A   B   C   D   E   p 
    # labels        = /s A   B   C   D   E   X
    # insert_tokens = /s A m B m C m D m E m   
    # insert_labels = /s A B C C D D E E X         
    # insert_mask   =  1 1 1 1 1 1 1 1 1 1 0  
    # inset_position=  0 1 2 2 3 3 4 4 5 5 6  
    # ex: /s A B C D E insert_token_num=2
    #               start----
    #                        |
    #                 /s A     B     C     D    E     p
    # insert_tokens = /s A m m B m m C m m D m m
    # insert_labels = /s A B C D C D E D E X  
    # insert_mask   =  1 1 1 1 1 1 1 1 1 1 1 1 0
    # inset_position=  0 1 2 3 2 3 4 3 4 5 4 5 6   
    
    if input_noise_ids is None:
        input_noise_ids = input_ids
    
    batch_size, seq_length = input_ids.shape[:2]    
    device = input_ids.device 
    labels = torch.where(input_ids != pad_id, input_ids, torch.tensor(-100))
    
    insert_tokens = input_noise_ids[:,:1]
    insert_labels = labels[:,:2]
    insert_mask = attention_mask[:,:1]
    insert_special_token = torch.full_like(input_ids[:,:1].repeat(1, insert_token_num), insert_token_id)
    ingnore_label = torch.full_like(labels[:,:1], torch.tensor(-100))
    
    if position_ids is None:
        device = input_ids.device 
        position_ids = torch.arange(
            0 , seq_length , dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)            

    insert_position = position_ids[:,:1]
    
    
    # for i ,(tokens, masks) in enumerate(zip(input_ids, attention_mask)):
    for j in range(2, seq_length-insert_token_num) :
        if sequence_label_ingnore :
            insert_labels = torch.cat((insert_labels, ingnore_label ,labels[:,j+1:j+insert_token_num+1]),dim=1)
        else:
            insert_labels = torch.cat((insert_labels,labels[:,j:j+insert_token_num+1]),dim=1)
    insert_labels = torch.cat((insert_labels, labels[:,-insert_token_num:]),dim=1)
        
        
    for j in range(1, seq_length-insert_token_num) :    
        insert_position = torch.cat((insert_position, position_ids[:,j:j+insert_token_num+1]),dim=1)
        insert_mask = torch.cat((insert_mask,attention_mask[:,j:j+insert_token_num+1]),dim=1)
        insert_tokens = torch.cat((insert_tokens,input_noise_ids[:,j].view(batch_size,1), insert_special_token),dim=1)
                   
    return  insert_tokens.detach(), insert_mask.detach(), insert_labels.detach(), insert_position.detach(), 
        
        
        
    