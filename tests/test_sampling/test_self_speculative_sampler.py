"""Unit tests for SelfSpeculativeSampler."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import initialize, compose
from unittest.mock import MagicMock, patch

from discrete_diffusion.sampling.self_speculative import SelfSpeculativeSampler


@pytest.fixture
def sampler():
    """Create SelfSpeculativeSampler using Hydra configs."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "sampling=self_speculative",
            "sampling.steps=10",
        ])
        return SelfSpeculativeSampler(cfg)


@pytest.fixture
def mock_model():
    """Create a mock model with all required interfaces for generate()."""
    model = MagicMock()
    model.num_tokens = 16
    model.device = torch.device("cpu")
    model.mask_id = 50256
    
    # Mock tokenizer
    model.tokenizer = MagicMock()
    model.tokenizer.bos_token_id = 50257
    
    # Mock prior_sample to return mask tokens
    def prior_sample(num_samples, num_tokens):
        return torch.full((num_samples, num_tokens), model.mask_id, dtype=torch.long)
    model.prior_sample = prior_sample
    
    # Mock noise schedule
    model.noise = MagicMock()
    model.noise.alpha_t = lambda t: torch.ones_like(t) * 0.5
    
    # Mock _sigma_from_alphat
    model._sigma_from_alphat = lambda alpha_t: torch.ones_like(alpha_t) * 0.5
    
    # Mock backbone with _draft_forward and _target_forward
    vocab_size = 50258
    hidden_size = 64
    
    def draft_forward(x, sigma, return_hidden_states=False):
        batch_size, seq_len = x.shape
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        if return_hidden_states:
            return hidden_states, logits
        return logits
    
    def target_forward(x0_draft, hidden_states, sigma, permutation):
        batch_size, seq_len = x0_draft.shape
        return torch.randn(batch_size, seq_len, vocab_size)
    
    model.backbone = MagicMock()
    model.backbone._draft_forward = draft_forward
    model.backbone._target_forward = target_forward
    
    return model


def test_sample_permutation(sampler):
    """Test permutation is a valid permutation of sequence length."""
    seq_len = 32
    device = torch.device("cpu")
    
    perm = sampler._sample_permutation(seq_len, device)
    
    assert perm.shape == (seq_len,)
    assert perm.dtype == torch.long
    # Check it's a valid permutation (contains all indices 0..seq_len-1)
    assert torch.all(torch.sort(perm)[0] == torch.arange(seq_len, device=device))


def test_permutation_random(sampler):
    """Test permutation is random (different calls give different results)."""
    seq_len = 32
    device = torch.device("cpu")
    
    perm1 = sampler._sample_permutation(seq_len, device)
    perm2 = sampler._sample_permutation(seq_len, device)
    
    # Different calls should give different permutations (with high probability)
    # This is probabilistic, but very unlikely to be identical
    assert not torch.equal(perm1, perm2)


def test_generate_sets_permutation(sampler, mock_model):
    """Test that generate() sets _permutation with correct shape."""
    num_samples = 4
    num_steps = 3
    eps = 0.001
    
    # Run generate
    samples = sampler.generate(
        mock_model, 
        num_samples=num_samples, 
        num_steps=num_steps, 
        eps=eps, 
        inject_bos=True
    )
    
    # Check permutation was set
    assert sampler._permutation is not None
    assert sampler._permutation.shape == (num_samples, mock_model.num_tokens)
    
    # Each row should be a valid permutation
    for i in range(num_samples):
        perm = sampler._permutation[i]
        sorted_perm, _ = torch.sort(perm)
        expected = torch.arange(mock_model.num_tokens)
        assert torch.equal(sorted_perm, expected), f"Row {i} is not a valid permutation"


def test_generate_output_shape(sampler, mock_model):
    """Test that generate() produces output with correct shape."""
    num_samples = 2
    num_steps = 5
    eps = 0.001
    
    samples = sampler.generate(
        mock_model, 
        num_samples=num_samples, 
        num_steps=num_steps, 
        eps=eps, 
        inject_bos=True
    )
    
    assert samples.shape == (num_samples, mock_model.num_tokens)


def test_generate_with_bos_injection(sampler, mock_model):
    """Test that BOS token is injected when inject_bos=True."""
    num_samples = 2
    num_steps = 3
    eps = 0.001
    
    samples = sampler.generate(
        mock_model, 
        num_samples=num_samples, 
        num_steps=num_steps, 
        eps=eps, 
        inject_bos=True
    )
    
    # First token should be BOS in all samples
    assert torch.all(samples[:, 0] == mock_model.tokenizer.bos_token_id)


def test_verify_and_mask_all_accepted(sampler, mock_model):
    """Test _verify_and_mask when all tokens are accepted."""
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    
    # Set up sampler state
    sampler._permutation = torch.stack([
        torch.randperm(seq_len) for _ in range(batch_size)
    ])
    
    # Create input with some masks
    x = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    x[:, 4:] = mock_model.mask_id  # Last 4 positions are masks
    
    # Create draft with high probability tokens
    x0_draft = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    
    # Create probs where target = draft (100% acceptance)
    draft_probs = torch.rand(batch_size, seq_len, vocab_size)
    draft_probs = F.softmax(draft_probs, dim=-1)
    target_probs = draft_probs.clone()  # Same probs = accept_prob = 1.0
    
    # Run verify_and_mask
    x_out = sampler._verify_and_mask(mock_model, x, draft_probs, target_probs, x0_draft)
    
    # All masked positions should be filled with draft tokens
    is_mask = (x == mock_model.mask_id)
    assert torch.all(x_out[is_mask] == x0_draft[is_mask])


def test_verify_and_mask_handles_no_masks(sampler, mock_model):
    """Test _verify_and_mask when there are no mask tokens."""
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    
    # Set up sampler state
    sampler._permutation = torch.stack([
        torch.randperm(seq_len) for _ in range(batch_size)
    ])
    
    # Create input with NO masks (all clean tokens)
    x = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    x0_draft = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    
    draft_probs = F.softmax(torch.rand(batch_size, seq_len, vocab_size), dim=-1)
    target_probs = F.softmax(torch.rand(batch_size, seq_len, vocab_size), dim=-1)
    
    # Run verify_and_mask - should not crash
    x_out = sampler._verify_and_mask(mock_model, x, draft_probs, target_probs, x0_draft)
    
    # Output should be identical to input (no masks to fill)
    assert torch.equal(x_out, x)


def test_verify_and_mask_clamps_rejection_index(sampler, mock_model):
    """Test that out-of-bounds rejection indices are handled correctly."""
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    
    # Set up sampler state
    sampler._permutation = torch.stack([
        torch.randperm(seq_len) for _ in range(batch_size)
    ])
    
    # Create input with all masks
    x = torch.full((batch_size, seq_len), mock_model.mask_id, dtype=torch.long)
    x0_draft = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    
    # Create probs that will cause all acceptances (no rejection)
    draft_probs = F.softmax(torch.rand(batch_size, seq_len, vocab_size), dim=-1)
    target_probs = draft_probs.clone()  # 100% acceptance
    
    # This should not crash even with first_rejection_idx == seq_len
    x_out = sampler._verify_and_mask(mock_model, x, draft_probs, target_probs, x0_draft)
    
    # All positions should be filled (all accepted)
    assert torch.all(x_out != mock_model.mask_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

