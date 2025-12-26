"""End-to-end test for self-speculative MDLM."""

import pytest
import torch
from hydra import initialize, compose


@pytest.mark.slow
def test_self_speculative_e2e():
    """Test full training and sampling pipeline."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "model=self_speculative_dit",
            "model.length=32",
            "model.n_blocks=2",
            "model.hidden_size=64",
            "model.cond_dim=64", 
            "model.n_heads=4",
            "algo=self_speculative_mdlm",
            "sampling=self_speculative",
            "data=openwebtext-split",
            "training.ema=0.0"
        ])
        
        from discrete_diffusion.data import get_tokenizer
        from discrete_diffusion.algorithms.self_speculative_mdlm import SelfSpeculativeMDLM
        
        tokenizer = get_tokenizer(cfg)
        model = SelfSpeculativeMDLM(cfg, tokenizer)
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, len(tokenizer), (batch_size, seq_len))
        attention_mask = torch.ones_like(x)
        
        # Simulate training step
        model.train()
        loss = model._loss(x, attention_mask)
        assert loss.loss.item() > 0
        
        # Test sampling
        model.eval()
        with torch.no_grad():
            samples = model.generate_samples(num_samples=2, num_steps=5)
        
        assert samples.shape == (2, seq_len)
        assert (samples >= 0).all()
        assert (samples < len(tokenizer)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

