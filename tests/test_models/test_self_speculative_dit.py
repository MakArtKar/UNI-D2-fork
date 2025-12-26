"""Unit tests for SelfSpeculativeDIT model."""

import pytest
import torch
from hydra import initialize, compose


@pytest.fixture
def model():
    """Create SelfSpeculativeDIT model using Hydra configs."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "model=self_speculative_dit",
            "model.length=32",
            "model.n_blocks=2",
            "model.hidden_size=64",
            "model.cond_dim=64",
            "model.n_heads=4",
            "algo=mdlm",
            "data=openwebtext-split"
        ])
        
        from discrete_diffusion.models.self_speculative_dit import SelfSpeculativeDIT
        
        vocab_size = 100
        return SelfSpeculativeDIT(cfg, vocab_size), cfg


def test_model_initialization(model):
    """Test model initializes with all components."""
    m, cfg = model
    assert hasattr(m, 'causal_layers')
    assert hasattr(m, 'target_head')
    assert hasattr(m, 'causal_input_proj')
    assert hasattr(m, 'pos_embedding')
    assert len(m.causal_layers) == 1


def test_draft_mode(model):
    """Test draft mode forward."""
    m, cfg = model
    x = torch.randint(0, 100, (2, 32))
    sigma = torch.rand(2)
    logits = m(x, sigma, mode="draft")
    assert logits.shape == (2, 32, 100)


def test_draft_mode_with_hidden_states(model):
    """Test draft mode with hidden states."""
    m, cfg = model
    x = torch.randint(0, 100, (2, 32))
    sigma = torch.rand(2)
    hidden, logits = m(x, sigma, mode="draft", return_hidden_states=True)
    assert hidden.shape == (2, 32, 64)
    assert logits.shape == (2, 32, 100)


def test_target_mode(model):
    """Test target mode forward."""
    m, cfg = model
    x = torch.randint(0, 100, (2, 32))
    sigma = torch.rand(2)
    hidden = torch.randn(2, 32, 64)
    perm = torch.stack([torch.randperm(32) for _ in range(2)])
    
    logits = m(x, sigma, mode="target", x0_draft=x, 
               draft_hidden_states=hidden, permutation=perm)
    assert logits.shape == (2, 32, 100)


def test_full_mode(model):
    """Test full mode forward."""
    m, cfg = model
    m.mask_id = 0
    x = torch.randint(0, 100, (2, 32))
    sigma = torch.rand(2)
    logits = m(x, sigma, mode="full")
    assert logits.shape == (2, 32, 100)


def test_generate_permutation(model):
    """Test permutation generation puts clean tokens first."""
    m, cfg = model
    x = torch.randint(1, 100, (2, 32))
    x[:, 10:20] = 0  # mask_id = 0
    perm = m._generate_permutation(x, mask_id=0)
    
    assert perm.shape == (2, 32)
    
    # Check clean positions come first
    for b in range(2):
        clean_count = (x[b] != 0).sum().item()
        for i in range(clean_count):
            pos = perm[b, i].item()
            assert x[b, pos] != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

