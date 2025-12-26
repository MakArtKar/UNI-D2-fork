"""Self-Speculative MDLM algorithm - trains target model while freezing draft."""

from .mdlm import MDLM


class SelfSpeculativeMDLM(MDLM):
    """Self-Speculative MDLM - freezes draft, trains only target components."""
    
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        
        self.backbone.mask_id = self.mask_id
        
        # Freeze ALL parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.noise.parameters():
            param.requires_grad = False
        
        # Unfreeze only target components using _get_parameters()
        for param in self._get_parameters():
            param.requires_grad = True
    
    def _get_parameters(self):
        """Return only trainable parameters."""
        trainable = []
        trainable.extend(self.backbone.causal_input_proj.parameters())
        trainable.extend(self.backbone.causal_layers.parameters())
        trainable.extend(self.backbone.target_head.parameters())
        trainable.extend(self.backbone.pos_embedding.parameters())
        return iter(trainable)
