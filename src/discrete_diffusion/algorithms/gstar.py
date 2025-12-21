"""GStar: Training a remasker to detect MDLM prediction errors."""

import torch
import torch.nn.functional as F

from .mdlm import MDLM
from ..forward_process.utils import sample_categorical
from ..models.common import DDiTFinalLayer


class GStar(MDLM):
    """GStar meta-learning algorithm.
    
    Freezes a pretrained MDLM backbone and trains a binary classifier
    (remasker_head) to predict whether MDLM's sampled prediction differs
    from the ground truth.
    
    Training flow:
    1. Standard MDLM forward: xt -> log_x_theta
    2. Sample: sampled_x0 = sample_categorical(exp(log_x_theta))
    3. Remasker forward: sampled_x0 -> hidden_states -> remasker_logits
    4. Loss: CE(remasker_logits, sampled_x0 != x0)
    """
    
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        
        # Freeze MDLM backbone and noise schedule
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.noise.parameters():
            param.requires_grad = False
        
        # Initialize remasker head (binary classification)
        hidden_size = config.model.hidden_size
        cond_dim = config.model.cond_dim
        adaLN = not config.algo.causal_attention
        
        self.remasker_head = DDiTFinalLayer(
            hidden_size=hidden_size,
            out_channels=2,  # Binary: correct (0) vs error (1)
            cond_dim=cond_dim,
            adaLN=adaLN
        )
        
    def _get_parameters(self):
        """Only return remasker_head parameters for optimization."""
        return self.remasker_head.parameters()
    
    def _remasker_forward(self, sampled_x0, sigma):
        """Forward pass that returns remasker logits.
        
        Takes MDLM's sampled predictions, passes them through the frozen
        backbone to get hidden states, then through the remasker head.
        
        Args:
            sampled_x0: Sampled tokens from MDLM [batch, seq_len]
            sigma: Noise level [batch, seq_len] or [batch]
        
        Returns:
            Tensor: Remasker logits [batch, seq_len, 2]
        """
        # Process sigma same way as parent class
        sigma_processed = self._process_sigma(sigma)
        
        # Get hidden states from frozen backbone by passing sampled_x0 through it
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = self.backbone(sampled_x0, sigma_processed, return_hidden_states=True)
        
        # Get time conditioning for remasker head (if needed)
        if self.backbone.causal:
            t_cond = None
        else:
            t_cond = F.silu(self.backbone.sigma_map(sigma_processed))
        
        # Apply remasker head (trainable)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            remasker_logits = self.remasker_head(hidden_states, c=t_cond)
        
        return remasker_logits
    
    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        """Compute remasker CE loss instead of MDLM NLL.
        
        Overrides parent nll_per_token to:
        1. Sample x0 from MDLM log probabilities
        2. Get remasker predictions on sampled_x0
        3. Compute binary CE loss (no rescheduling)
        
        Args:
            log_x_theta: Log-probabilities from MDLM [batch, seq_len, vocab]
            xt: Noisy input (not used in GStar)
            x0: Ground truth tokens [batch, seq_len]
            alpha_t: Schedule value (not used - no rescheduling)
            dalpha_t: Schedule derivative (not used - no rescheduling)
            low_var: Low variance flag (not used)
        
        Returns:
            Tensor: Raw per-token CE loss [batch, seq_len]
        """
        # Ignore xt, alpha_t, dalpha_t, low_var - we don't need them
        
        # Sample x0 from MDLM predictions
        with torch.no_grad():
            sampled_x0 = sample_categorical(log_x_theta.exp())
        
        # Compute sigma for remasker (reconstruct from alpha_t)
        sigma = self._sigma_from_alphat(alpha_t)
        
        # Get remasker predictions
        remasker_logits = self._remasker_forward(sampled_x0, sigma)
        
        # Compute binary targets: 1 if error, 0 if correct
        targets = (sampled_x0 != x0).long()  # [batch, seq_len]
        
        # Cross-entropy loss per token (raw, no rescheduling)
        # remasker_logits: [batch, seq_len, 2]
        # targets: [batch, seq_len]
        ce_loss = F.cross_entropy(
            remasker_logits.transpose(1, 2),  # [batch, 2, seq_len]
            targets,
            reduction='none'
        )  # [batch, seq_len]
        
        # Return raw CE loss (no weighting by dalpha_t/(1-alpha_t))
        return ce_loss

