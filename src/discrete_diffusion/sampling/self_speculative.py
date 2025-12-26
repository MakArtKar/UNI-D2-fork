"""Self-Speculative sampler with speculative decoding verification."""

from __future__ import annotations

import math
from functools import partial
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from .absorbing import AbsorbingSampler


class SelfSpeculativeSampler(AbsorbingSampler):
    """Sampler implementing self-speculative decoding for MDLM.
    
    Supports three modes:
    - "spec_decoding": Full speculative decoding with draft + target verification
    - "draft": Use draft model only, following MDLM schedule
    - "target": Autoregressive decoding, 1 token at a time in permutation order
    
    Window schedule (for spec_decoding mode):
    - None: No window limit, verify all tokens at once
    - "linear": Window size increases linearly from 1 to num_masks
    - "cosine": Window size increases following cosine schedule (slower start, faster end)
    
    From the paper (https://arxiv.org/abs/2510.03929):
    - Early steps: high uncertainty → small window (careful, 1 token at a time)
    - Late steps: more context → large window (can denoise more tokens)
    
    See Appendix D of the paper for details.
    """
    
    def __init__(self, config, forward_process=None, 
                 mode: Literal["spec_decoding", "draft", "target"] = "spec_decoding"):
        super().__init__(config, forward_process)
        self._permutation = None
        self._draft_hidden_states = None  # Cache for target forward
        
        # Get mode from config if not explicitly passed
        self.mode = config.sampling.get("mode", mode)
        
        # Window schedule for limiting tokens revealed per step
        self.window_schedule: Optional[str] = config.sampling.get("window_schedule", None)
        
        # Cosine window parameter Δτ from paper (Table 3)
        # Controls max tokens revealed per step:
        # - Δτ = 0.002 → ~1 draft/verify step (most AR-like)
        # - Δτ = 0.005 → ~2 draft/verify steps
        # - Δτ = 0.01  → ~3 draft/verify steps  
        # - Δτ = 0.083 → ~10 draft/verify steps (most parallel)
        # Default None means use full num_masks as max window
        self.cosine_delta_t: Optional[float] = config.sampling.get("cosine_delta_t", None)
        
        # Track generation progress for window calculation
        self._current_step = 0
        self._total_steps = 0
    
    def _sample_permutation(self, seq_len, device):
        """Sample permutation of length seq_len.
        
        Returns a random permutation of [0, 1, ..., seq_len-1].
        """
        perm = torch.randperm(seq_len, device=device)
        return perm
    
    def _get_window_size(self, num_masks: int) -> int:
        """Get the window size based on schedule and generation progress.
        
        The window limits how many tokens can be revealed in a single step,
        ensuring the speculative sampling guarantee holds within the window.
        
        From the paper (https://arxiv.org/abs/2510.03929):
        - Early steps: high uncertainty, many masks → small window (careful)
        - Late steps: more context, few masks → large window (can denoise more)
        
        Args:
            num_masks: Current number of masked tokens.
            
        Returns:
            Maximum number of tokens that can be revealed in this step.
        """
        if self.window_schedule is None or num_masks == 0:
            return num_masks
        
        if self._total_steps <= 1:
            return num_masks
        
        progress = self._current_step / (self._total_steps - 1)  # 0 to 1
        
        if self.window_schedule == "linear":
            # Linear schedule: window INCREASES from 1 to num_masks
            # Early steps (progress~0): window = 1 (careful, denoise 1 at a time)
            # Late steps (progress~1): window = num_masks (can denoise more)
            window = max(1, int(1 * (1 - progress) + num_masks * progress))
            return window
        
        elif self.window_schedule == "cosine":
            # Cosine schedule: slower start (more careful early), faster ramp-up late
            # Uses cosine interpolation: (1 - cos(π * progress)) / 2
            # At progress=0: cos(0)=1, factor=0, window=1
            # At progress=1: cos(π)=-1, factor=1, window=max_window
            # This keeps window small for longer at the start (more AR-like behavior)
            
            # Determine max window based on cosine_delta_t parameter
            if self.cosine_delta_t is not None:
                # delta_t controls max tokens to reveal per step
                # From paper Table 3: delta_t=0.002 → ~1 token, delta_t=0.083 → ~10 tokens
                # Scale: max_window = num_masks * delta_t * scale_factor
                # With scale_factor=50: delta_t=0.002 → 127*0.002*50 ≈ 12.7
                # We use a tighter scaling to match paper's behavior
                max_window = max(1, int(self._total_steps * self.cosine_delta_t))
            else:
                max_window = num_masks
            
            cosine_factor = (1 - math.cos(math.pi * progress)) / 2
            window = max(1, int(1 + (max_window - 1) * cosine_factor))
            return min(window, num_masks)  # Can't exceed actual masks
        
        # Default: no window limit
        return num_masks
    
    def _draft_forward_wrapper(self, model, x, sigma):
        """Wrapper for draft forward that caches hidden states.
        
        Returns log probabilities (log_softmax of logits) to match
        the expected interface of _sample_x0.
        """
        # Ensure sigma is 1D [batch] for _draft_forward
        if sigma.ndim == 2:
            sigma = sigma.squeeze(-1)
        elif sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        hidden_states, draft_logits = model.backbone._draft_forward(
            x, sigma, return_hidden_states=True
        )
        self._draft_hidden_states = hidden_states
        # Convert logits to log probabilities (what _sample_x0 expects)
        return F.log_softmax(draft_logits, dim=-1)
    
    def _get_permuted_tensors(self, x, x0_draft, mask_id):
        """Reorder tensors according to permutation.
        
        Args:
            x: Current sequence [B, S].
            x0_draft: Draft predictions [B, S].
            mask_id: Mask token ID.
            
        Returns:
            Tuple of (is_mask, is_mask_perm, x0_draft_perm, batch_idx, pos_indices).
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        is_mask = (x == mask_id)  # [B, S]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        is_mask_perm = is_mask[batch_idx, self._permutation]  # [B, S] in perm order
        x0_draft_perm = x0_draft[batch_idx, self._permutation]  # [B, S] in perm order
        
        return is_mask, is_mask_perm, x0_draft_perm, batch_idx, pos_indices
    
    def _calculate_first_mask_position(self, is_mask_perm):
        """Calculate the first mask position in permutation order for each batch.
        
        Args:
            is_mask_perm: Boolean mask indicating masked positions in perm order [B, S].
            
        Returns:
            first_mask_idx: Index of first mask position per batch [B].
        """
        batch_size, seq_len = is_mask_perm.shape
        
        # cumsum gives us indices where mask first appears
        mask_cumsum = is_mask_perm.cumsum(dim=1)  # [B, S]
        first_mask_idx = (mask_cumsum == 1).long().argmax(dim=1)  # [B]
        
        # If no masks, set to seq_len
        first_mask_idx = torch.where(
            is_mask_perm.any(dim=1),
            first_mask_idx,
            torch.full_like(first_mask_idx, seq_len)
        )
        
        return first_mask_idx
    
    def _calculate_acceptance_mask(self, draft_probs, target_probs, x0_draft_perm, 
                                   is_mask_perm, batch_idx):
        """Calculate acceptance mask for all tokens using speculative decoding formula.
        
        Args:
            draft_probs: Probabilities from draft model [B, S, V].
            target_probs: Probabilities from target model [B, S, V].
            x0_draft_perm: Draft tokens in permutation order [B, S].
            is_mask_perm: Boolean mask in permutation order [B, S].
            batch_idx: Batch indices for gathering [B, S].
            
        Returns:
            acceptance_mask: Boolean mask of accepted positions in perm order [B, S].
        """
        batch_size, seq_len = is_mask_perm.shape
        device = is_mask_perm.device
        eps = 1e-10
        
        # Gather probabilities for draft tokens at each position
        draft_token_probs_perm = draft_probs[batch_idx, self._permutation, x0_draft_perm]
        target_token_probs_perm = target_probs[batch_idx, self._permutation, x0_draft_perm]
        
        # Calculate acceptance probabilities: target_prob / draft_prob
        # No need to clamp to 1.0 - values > 1 will always pass u <= accept_probs check
        accept_probs = target_token_probs_perm / torch.clamp(draft_token_probs_perm, min=eps)
        
        # Sample random values and compare
        u = torch.rand(batch_size, seq_len, device=device)
        acceptance_mask = (u <= accept_probs)
        
        # Only consider masked positions for acceptance
        acceptance_mask = acceptance_mask & is_mask_perm
        
        return acceptance_mask
    
    def _find_first_rejection(self, acceptance_mask, is_mask_perm, first_mask_idx, pos_indices):
        """Find the first unaccepted position after the first mask in permutation order.
        
        Args:
            acceptance_mask: Boolean mask of accepted positions in perm order [B, S].
            is_mask_perm: Boolean mask in permutation order [B, S].
            first_mask_idx: Index of first mask position per batch [B].
            pos_indices: Position indices [B, S].
            
        Returns:
            first_rejection_idx: Index of first rejection per batch in perm order [B].
        """
        batch_size, seq_len = is_mask_perm.shape
        
        # Find positions that are after first mask and not accepted
        after_first_mask = pos_indices >= first_mask_idx.unsqueeze(1)
        unaccepted = after_first_mask & is_mask_perm & (~acceptance_mask)
        
        # Find first unaccepted position using argmax
        unaccepted_long = torch.where(
            after_first_mask,
            unaccepted.long(),
            torch.full_like(unaccepted, -1, dtype=torch.long)
        )
        first_rejection_idx = unaccepted_long.argmax(dim=1)
        
        # If no unaccepted positions, set to seq_len
        has_rejection = unaccepted.any(dim=1)
        first_rejection_idx = torch.where(
            has_rejection,
            first_rejection_idx,
            torch.full_like(first_rejection_idx, seq_len)
        )
        
        return first_rejection_idx
    
    def _apply_token_updates(self, x, x0_draft, draft_probs, target_probs, is_mask,
                             acceptance_mask, first_rejection_idx, batch_idx, 
                             pos_indices, mask_id):
        """Apply token updates: accepted tokens, first rejection token, and mask remaining.
        
        Args:
            x: Current sequence [B, S].
            x0_draft: Draft predictions [B, S].
            draft_probs: Probabilities from draft model [B, S, V].
            target_probs: Probabilities from target model [B, S, V].
            is_mask: Boolean mask in original order [B, S].
            acceptance_mask: Boolean mask of accepted positions in perm order [B, S].
            first_rejection_idx: Index of first rejection per batch in perm order [B].
            batch_idx: Batch indices [B, S].
            pos_indices: Position indices [B, S].
            mask_id: Mask token ID.
            
        Returns:
            x_out: Updated sequence [B, S].
        """
        batch_size, seq_len = x.shape
        device = x.device
        vocab_size = draft_probs.shape[-1]
        eps = 1e-10
        
        x_out = x.clone()
        
        # --- Step 1: Replace accepted masked positions with x0_draft ---
        # Reorder acceptance_mask back to original order
        inv_perm = torch.zeros_like(self._permutation)
        inv_perm[batch_idx, self._permutation] = pos_indices
        acceptance_mask_orig = acceptance_mask[batch_idx, inv_perm]
        
        x_out = torch.where(acceptance_mask_orig & is_mask, x0_draft, x_out)
        
        # --- Step 2: Resample and replace first rejection position ---
        # Calculate adjusted distribution: max(0, target - draft)
        adjusted_probs = torch.clamp(target_probs - draft_probs, min=0)
        adjusted_sums = adjusted_probs.sum(dim=-1, keepdim=True)
        
        adjusted_probs_normalized = torch.where(
            adjusted_sums > eps,
            adjusted_probs / adjusted_sums,
            target_probs
        )
        
        # Resample tokens from adjusted distribution
        x0_target = sample_categorical(adjusted_probs_normalized)
        
        # Create mask for first rejection position only
        safe_rejection_idx = torch.clamp(first_rejection_idx, max=seq_len - 1)
        first_rejection_pos_orig = self._permutation.gather(1, safe_rejection_idx.unsqueeze(1)).squeeze(1)
        
        first_rejection_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        valid_rejection = first_rejection_idx < seq_len
        first_rejection_mask.scatter_(1, first_rejection_pos_orig.unsqueeze(1), 
                                       valid_rejection.unsqueeze(1))
        
        x_out = torch.where(first_rejection_mask & is_mask, x0_target, x_out)
        
        # --- Step 3: Mask all positions after first rejection ---
        after_rejection_in_perm = pos_indices > first_rejection_idx.unsqueeze(1)
        has_rejection_expanded = (first_rejection_idx < seq_len).unsqueeze(1)
        after_rejection_in_perm = after_rejection_in_perm & has_rejection_expanded
        
        # Scatter to original positions
        after_rejection_mask = torch.zeros_like(is_mask, dtype=torch.long)
        after_rejection_mask.scatter_(1, self._permutation, after_rejection_in_perm.long())
        after_rejection_mask = after_rejection_mask.bool()
        
        x_out = torch.where(after_rejection_mask & is_mask, 
                           torch.full_like(x_out, mask_id), x_out)
        
        return x_out
    
    def _verify_and_mask(self, model, x, draft_probs, target_probs, x0_draft):
        """Verify tokens and mask unaccepted positions (vectorized).
        
        With window schedule enabled, only considers tokens within the window
        for verification. This ensures the speculative sampling guarantee holds
        within the window (see Appendix D of https://arxiv.org/abs/2510.03929).
        
        On rejection at position i (permutation order):
        - Resample token at i from adjusted distribution
        - Keep all accepted tokens (positions < i)
        - Keep resampled token at i
        - Mask all positions > i (or outside window)
        """
        batch_size, seq_len = x.shape
        
        # 1) Get permuted tensors
        is_mask, is_mask_perm, x0_draft_perm, batch_idx, pos_indices = \
            self._get_permuted_tensors(x, x0_draft, model.mask_id)
        
        # 2) Calculate first mask position in permutation order
        first_mask_idx = self._calculate_first_mask_position(is_mask_perm)
        
        # 3) Apply window limit if enabled
        # Window limits how many masked positions we consider for verification
        num_masks = is_mask_perm.sum(dim=1)  # [B]
        max_num_masks = num_masks.max().item()
        window_size = self._get_window_size(max_num_masks)
        
        if self.window_schedule is not None and window_size < max_num_masks:
            # Create window mask: only consider first `window_size` masked positions
            # in permutation order
            mask_cumsum = is_mask_perm.cumsum(dim=1)  # [B, S]
            within_window = mask_cumsum <= window_size  # [B, S]
            
            # Update is_mask_perm to only include positions within window
            is_mask_perm_windowed = is_mask_perm & within_window
        else:
            is_mask_perm_windowed = is_mask_perm
        
        # 4) Calculate acceptance mask for tokens within window
        acceptance_mask = self._calculate_acceptance_mask(
            draft_probs, target_probs, x0_draft_perm, is_mask_perm_windowed, batch_idx
        )
        
        # 5) Find first unaccepted position (within window)
        first_rejection_idx = self._find_first_rejection(
            acceptance_mask, is_mask_perm_windowed, first_mask_idx, pos_indices
        )
        
        # 6) Apply token updates: accepted tokens, first rejection, mask remaining
        x_out = self._apply_token_updates(
            x, x0_draft, draft_probs, target_probs, is_mask,
            acceptance_mask, first_rejection_idx, batch_idx, pos_indices, model.mask_id
        )
        
        return x_out
    
    def compute_posterior(self, model, x, t, dt, p_x0=None, noise_removal_step=False):
        """Compute posterior based on sampling mode.
        
        Modes:
        - "spec_decoding": Full speculative decoding with draft + target verification
        - "draft": Use draft model only, following MDLM schedule  
        - "target": Autoregressive decoding, 1 token at a time in permutation order
        """
        if self.mode == "draft":
            return self._compute_posterior_draft(model, x, t, dt, p_x0, noise_removal_step)
        elif self.mode == "target":
            return self._compute_posterior_target(model, x, t, dt, p_x0, noise_removal_step)
        else:  # spec_decoding
            return self._compute_posterior_spec(model, x, t, dt, p_x0, noise_removal_step)
    
    def _compute_posterior_draft(self, model, x, t, dt, p_x0, noise_removal_step):
        """Draft mode: use draft model only, following MDLM schedule."""
        draft_forward = partial(self._draft_forward_wrapper, model)
        return super().compute_posterior(
            model, x, t, dt, p_x0=p_x0, 
            noise_removal_step=noise_removal_step,
            forward_method=draft_forward
        )
    
    def _get_draft_and_target_probs(self, model, x, t):
        """Get draft and target probabilities (shared by spec and target modes).
        
        Returns:
            draft_probs: Probabilities from draft model [B, S, V].
            target_probs: Probabilities from target model [B, S, V].
            x0_draft: Sampled tokens from draft distribution [B, S].
        """
        # Get draft predictions (caches hidden states)
        draft_forward = partial(self._draft_forward_wrapper, model)
        draft_probs, x0_draft = self._sample_x0(
            model, x, t, p_x0=None, forward_method=draft_forward
        )
        
        # Get target logits using cached hidden states
        alpha_t = model.noise.alpha_t(t)
        sigma = model._sigma_from_alphat(alpha_t)
        if sigma.ndim == 2:
            sigma = sigma.squeeze(-1)
        elif sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        
        target_logits = model.backbone._target_forward(
            x0_draft, self._draft_hidden_states, sigma, self._permutation
        )
        target_probs = F.softmax(target_logits, dim=-1)
        
        return draft_probs, target_probs, x0_draft
    
    def _denoise_first_mask_only(self, x, target_probs, mask_id):
        """Denoise only the first masked position in permutation order.
        
        Used by target mode for AR decoding.
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Get mask info in permutation order
        is_mask = (x == mask_id)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        is_mask_perm = is_mask[batch_idx, self._permutation]
        
        # Find first mask position
        first_mask_idx = self._calculate_first_mask_position(is_mask_perm)
        
        # Sample from target distribution
        x0_target = sample_categorical(target_probs)
        
        # Create mask for first mask position only (in original order)
        safe_first_mask_idx = torch.clamp(first_mask_idx, max=seq_len - 1)
        first_mask_pos_orig = self._permutation.gather(1, safe_first_mask_idx.unsqueeze(1)).squeeze(1)
        
        has_mask = is_mask.any(dim=1)
        first_mask_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        first_mask_mask.scatter_(1, first_mask_pos_orig.unsqueeze(1), has_mask.unsqueeze(1))
        
        return torch.where(first_mask_mask, x0_target, x)
    
    def _compute_posterior_target(self, model, x, t, dt, p_x0, noise_removal_step):
        """Target mode: AR decoding, 1 token at a time in permutation order."""
        _, target_probs, _ = self._get_draft_and_target_probs(model, x, t)
        x_out = self._denoise_first_mask_only(x, target_probs, model.mask_id)
        return None, x_out
    
    def _compute_posterior_spec(self, model, x, t, dt, p_x0, noise_removal_step):
        """Speculative decoding mode: draft + target verification."""
        draft_probs, target_probs, x0_draft = self._get_draft_and_target_probs(model, x, t)
        x_out = self._verify_and_mask(model, x, draft_probs, target_probs, x0_draft)
        
        # Increment step counter for window calculation
        self._current_step += 1
        
        return None, x_out
    
    @torch.no_grad()
    def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
        """Generate samples - sample permutation once then call parent generate."""
        # Sample permutation with model.length length (not based on mask positions)
        # Permutation is [batch, seq_len] where each row is a random permutation
        seq_len = model.num_tokens
        device = model.device
        self._permutation = torch.stack([
            self._sample_permutation(seq_len, device) 
            for _ in range(num_samples)
        ])
        
        # Track step progress for window calculation
        self._current_step = 0
        self._total_steps = num_steps
        
        # Call parent's generate (which will call our compute_posterior)
        return super().generate(model=model, num_samples=num_samples, 
                               num_steps=num_steps, eps=eps, inject_bos=inject_bos)

