"""Unit tests for SelfSpeculativeMDLM algorithm."""

import pytest
import torch
from hydra import initialize, compose


@pytest.fixture
def model():
    """Create SelfSpeculativeMDLM model using Hydra configs."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "model=self_speculative_dit",
            "model.length=32",
            "model.n_blocks=2",
            "model.hidden_size=64",
            "model.cond_dim=64",
            "model.n_heads=4",
            "algo=self_speculative_mdlm",
            "data=openwebtext-split",
            "training.ema=0.0"
        ])
        
        from discrete_diffusion.data import get_tokenizer
        from discrete_diffusion.algorithms.self_speculative_mdlm import SelfSpeculativeMDLM
        
        tokenizer = get_tokenizer(cfg)
        return SelfSpeculativeMDLM(cfg, tokenizer), cfg


def test_frozen_parameters(model):
    """Test that draft components are frozen."""
    algo, cfg = model
    
    # Backbone blocks should be frozen
    for param in algo.backbone.blocks.parameters():
        assert not param.requires_grad
    
    # Output layer (draft head) should be frozen
    for param in algo.backbone.output_layer.parameters():
        assert not param.requires_grad
    
    # Noise schedule should be frozen
    for param in algo.noise.parameters():
        assert not param.requires_grad


def test_trainable_target(model):
    """Test that target components are trainable."""
    algo, cfg = model
    
    # Causal layers should be trainable
    for param in algo.backbone.causal_layers.parameters():
        assert param.requires_grad
    
    # Target head should be trainable
    for param in algo.backbone.target_head.parameters():
        assert param.requires_grad
    
    # Causal input projection should be trainable
    for param in algo.backbone.causal_input_proj.parameters():
        assert param.requires_grad
    
    # Position embedding should be trainable
    for param in algo.backbone.pos_embedding.parameters():
        assert param.requires_grad


def test_get_parameters(model):
    """Test _get_parameters returns only trainable params."""
    algo, cfg = model
    
    params = list(algo._get_parameters())
    
    # Should have parameters from causal components only
    assert len(params) > 0
    
    # All returned params should require grad
    for p in params:
        assert p.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

