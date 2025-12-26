"""Self-Speculative DiT model for speculative decoding in masked diffusion."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from transformers import GPT2LMHeadModel

from .dit import DIT
from .common import DDiTBlockCausal, DDiTFinalLayer
from ..forward_process.utils import sample_categorical


class SelfSpeculativeDIT(DIT):
    """DiT extended with causal verification layers for self-speculative decoding.
    
    Forward modes:
    - "draft": Only draft forward (reuses parent DIT.forward)
    - "target": Only target forward (requires x0_draft, hidden_states, permutation)
    - "full": Both draft and target (default for training)
    """
    
    def __init__(self, config, vocab_size: int):
        super().__init__(config, vocab_size)
        
        dim = config.model.hidden_size
        n_heads = config.model.n_heads
        dropout = config.model.dropout
        cond_dim = config.model.cond_dim
        
        self.num_causal_layers = getattr(config.model, 'num_causal_layers', 1)
        self.rotary_dim = dim // n_heads
        
        # Input projection for causal layers
        causal_input_dim = 2 * dim + dim + 2 * self.rotary_dim
        self.causal_input_proj = nn.Linear(causal_input_dim, dim)
        
        # Causal transformer layers
        self.causal_layers = nn.ModuleList([
            DDiTBlockCausal(dim=dim, n_heads=n_heads, dropout=dropout)
            for _ in range(self.num_causal_layers)
        ])
        
        # Separate target head
        self.target_head = DDiTFinalLayer(
            hidden_size=dim, out_channels=vocab_size,
            cond_dim=cond_dim, adaLN=self.adaLN
        )
        
        # Position embedding - initialize from GPT2
        max_seq_len = getattr(config.model, 'length', 1024)
        self.pos_embedding = nn.Embedding(max_seq_len, self.rotary_dim)
        self._init_pos_embedding_from_gpt2(max_seq_len)
    
    def _init_pos_embedding_from_gpt2(self, max_seq_len: int):
        """Initialize position embeddings from GPT2 pretrained weights."""
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_pos_emb = gpt2.transformer.wpe.weight.data  # [1024, 768]
        
        src_len, src_dim = gpt2_pos_emb.shape
        tgt_len, tgt_dim = max_seq_len, self.rotary_dim
        
        init_emb = gpt2_pos_emb[:min(src_len, tgt_len), :min(src_dim, tgt_dim)]
        
        if tgt_len > src_len:
            padding = gpt2_pos_emb[-1:, :min(src_dim, tgt_dim)].expand(tgt_len - src_len, -1)
            init_emb = torch.cat([init_emb, padding], dim=0)
        
        if tgt_dim > src_dim:
            init_emb = F.pad(init_emb, (0, tgt_dim - src_dim))
        
        self.pos_embedding.weight.data.copy_(init_emb)
        del gpt2
    
    def _draft_forward(self, x, sigma, return_hidden_states: bool = False):
        """Draft forward - reuses parent DIT.forward()."""
        if return_hidden_states:
            # Get hidden states from parent (sigma used directly, no processing)
            hidden_states = super().forward(x, sigma, return_hidden_states=True)
            # Compute draft logits using output_layer (sigma used directly like parent DIT)
            if self.causal:
                t_cond = None
            else:
                t_cond = F.silu(self.sigma_map(sigma))
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                draft_logits = self.output_layer(hidden_states, c=t_cond)
            return hidden_states, draft_logits
        else:
            # Just return logits from parent (sigma used directly)
            return super().forward(x, sigma, return_hidden_states=False)
    
    def _generate_permutation(self, x, mask_id):
        """Generate permutation where clean tokens come before masked tokens."""
        batch_size, seq_len = x.shape
        device = x.device
        is_mask = (x == mask_id)
        
        permutations = []
        for b in range(batch_size):
            clean_pos = (~is_mask[b]).nonzero(as_tuple=True)[0]
            mask_pos = is_mask[b].nonzero(as_tuple=True)[0]
            perm_idx = torch.randperm(mask_pos.size(0), device=device)
            mask_pos_shuffled = mask_pos[perm_idx]
            perm = torch.cat([clean_pos, mask_pos_shuffled])
            permutations.append(perm)
        
        return torch.stack(permutations)
    
    def _build_causal_input(self, hidden_states, x0_draft, permutation):
        """Build input for causal layers.
        
        Returns:
            causal_input: Projected input for causal layers [B, S, D].
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        perm_expanded = permutation.unsqueeze(-1).expand(-1, -1, hidden_size)
        hs_perm = torch.gather(hidden_states, 1, perm_expanded)
        
        next_perm = torch.cat([permutation[:, 1:], permutation[:, :1]], dim=1)
        next_perm_expanded = next_perm.unsqueeze(-1).expand(-1, -1, hidden_size)
        hs_next_perm = torch.gather(hidden_states, 1, next_perm_expanded)
        hs_next_perm[:, -1, :] = 0
        
        x0_perm = torch.gather(x0_draft, 1, permutation)
        token_emb = self.vocab_embed(x0_perm)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb_current = self.pos_embedding(torch.gather(positions, 1, permutation))
        pos_emb_next = self.pos_embedding(torch.gather(positions, 1, next_perm))
        pos_emb_next[:, -1, :] = 0
        
        causal_input = torch.cat([
            hs_perm, hs_next_perm, token_emb, pos_emb_current, pos_emb_next
        ], dim=-1)
        
        return self.causal_input_proj(causal_input)
    
    def _target_forward(self, x0_draft, draft_hidden_states, sigma, permutation):
        """Target forward through causal layers + target head.
        
        Includes residual connection from draft hidden states to causal output,
        allowing causal layers to learn corrections rather than full representations.
        
        Key: Residual is added in ORIGINAL order to maintain position correspondence:
        - hidden_original[j] = causal output for original position j
        - draft_hidden_states[j] = draft hidden for original position j
        - Both correspond to the same position j
        """
        causal_input = self._build_causal_input(draft_hidden_states, x0_draft, permutation)
        
        # Use sigma directly like parent DIT does (no processing)
        if self.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.sigma_map(sigma))
        
        rotary_cos_sin = self.rotary_emb(causal_input)
        
        hidden = causal_input
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for layer in self.causal_layers:
                hidden = layer(hidden, rotary_cos_sin)
            
            # Reorder causal output back to original order BEFORE adding residual
            batch_size, seq_len, hidden_size = hidden.shape
            inv_perm = torch.zeros_like(permutation)
            batch_idx = torch.arange(batch_size, device=permutation.device).unsqueeze(1).expand_as(permutation)
            inv_perm[batch_idx, permutation] = torch.arange(seq_len, device=permutation.device).unsqueeze(0).expand(batch_size, -1)
            
            inv_perm_expanded = inv_perm.unsqueeze(-1).expand(-1, -1, hidden_size)
            hidden_original = torch.gather(hidden, 1, inv_perm_expanded)
            
            # Residual connection in ORIGINAL order - maintains position correspondence
            # hidden_original[j] corresponds to original position j
            # draft_hidden_states[j] corresponds to original position j
            hidden_with_residual = hidden_original + draft_hidden_states
            
            target_logits = self.target_head(hidden_with_residual, c=t_cond)
        
        # Already in original order - no need to reorder
        return target_logits
    
    def forward(self, x, sigma, mode: Literal["draft", "target", "full"] = "full",
                return_hidden_states: bool = False,
                x0_draft: Optional[torch.Tensor] = None,
                draft_hidden_states: Optional[torch.Tensor] = None,
                permutation: Optional[torch.Tensor] = None):
        """Forward pass with mode selection."""
        # Don't process sigma here - parent DIT handles it
        # Just ensure it's at least 1D for consistency
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        
        if mode == "draft":
            return self._draft_forward(x, sigma, return_hidden_states)
        
        elif mode == "target":
            assert x0_draft is not None and draft_hidden_states is not None and permutation is not None
            return self._target_forward(x0_draft, draft_hidden_states, sigma, permutation)
        
        else:  # full mode
            hidden_states, draft_logits = self._draft_forward(x, sigma, return_hidden_states=True)
            
            with torch.no_grad():
                draft_probs = F.softmax(draft_logits, dim=-1)
                x0_draft = sample_categorical(draft_probs)
            
            if permutation is None:
                mask_id = getattr(self, 'mask_id', self.vocab_size)
                permutation = self._generate_permutation(x, mask_id)
            
            target_logits = self._target_forward(x0_draft, hidden_states, sigma, permutation)
            
            if return_hidden_states:
                return target_logits, hidden_states, draft_logits
            return target_logits

