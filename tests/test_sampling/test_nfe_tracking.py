"""Unit tests for NFE (Number of Function Evaluations) tracking in samplers."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from hydra import initialize, compose

from discrete_diffusion.sampling.absorbing import AbsorbingSampler


@pytest.fixture
def sampler_with_nfe():
    """Create AbsorbingSampler with NFE tracking enabled."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "sampling=default",
            "sampling.steps=10",
            "sampling.predictor=ddpm",
            "++sampling.nfe_metric=True",
        ])
        return AbsorbingSampler(cfg)


@pytest.fixture
def sampler_without_nfe():
    """Create AbsorbingSampler without NFE tracking."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "sampling=default",
            "sampling.steps=10",
            "sampling.predictor=ddpm",
        ])
        return AbsorbingSampler(cfg)


@pytest.fixture
def mock_model():
    """Create a mock model for testing generate()."""
    model = MagicMock()
    model.num_tokens = 16
    model.device = torch.device("cpu")
    model.mask_id = 50256
    model.time_conditioning = False
    
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
    
    # Mock forward - returns random logits
    vocab_size = 50258
    def forward(x, sigma):
        batch_size, seq_len = x.shape
        return torch.randn(batch_size, seq_len, vocab_size)
    model.forward = forward
    
    return model


def test_nfe_tracking_returns_tuple_when_enabled(sampler_with_nfe, mock_model):
    """Test that generate returns (samples, mean_nfe) when nfe_metric=True."""
    result = sampler_with_nfe.generate(
        mock_model,
        num_samples=4,
        num_steps=5,
        eps=0.001,
        inject_bos=True
    )
    
    assert isinstance(result, tuple), "Should return tuple when nfe_metric=True"
    assert len(result) == 2, "Tuple should have 2 elements: (samples, mean_nfe)"
    
    samples, mean_nfe = result
    assert samples.shape == (4, mock_model.num_tokens)
    assert isinstance(mean_nfe, float)
    assert mean_nfe > 0


def test_nfe_tracking_returns_tensor_when_disabled(sampler_without_nfe, mock_model):
    """Test that generate returns just samples when nfe_metric=False."""
    result = sampler_without_nfe.generate(
        mock_model,
        num_samples=4,
        num_steps=5,
        eps=0.001,
        inject_bos=True
    )
    
    assert isinstance(result, torch.Tensor), "Should return Tensor when nfe_metric=False"
    assert result.shape == (4, mock_model.num_tokens)


def test_nfe_value_is_reasonable(sampler_with_nfe, mock_model):
    """Test that mean_nfe is within expected range (1 to num_steps+1)."""
    num_steps = 10
    
    samples, mean_nfe = sampler_with_nfe.generate(
        mock_model,
        num_samples=8,
        num_steps=num_steps,
        eps=0.001,
        inject_bos=True
    )
    
    # NFE should be between 1 (finished after first step) and num_steps+1 (noise removal)
    assert 1 <= mean_nfe <= num_steps + 1, f"mean_nfe {mean_nfe} out of range [1, {num_steps+1}]"


def test_nfe_early_completion():
    """Test NFE tracking when samples complete at different steps."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "sampling=default",
            "sampling.steps=10",
            "sampling.predictor=ddpm",
            "++sampling.nfe_metric=True",
        ])
        sampler = AbsorbingSampler(cfg)
    
    # Create a mock model that progressively unmasks positions
    model = MagicMock()
    model.num_tokens = 8
    model.device = torch.device("cpu")
    model.mask_id = 50256
    model.time_conditioning = False
    model.tokenizer = MagicMock()
    model.tokenizer.bos_token_id = 50257
    
    def prior_sample(num_samples, num_tokens):
        return torch.full((num_samples, num_tokens), model.mask_id, dtype=torch.long)
    model.prior_sample = prior_sample
    
    model.noise = MagicMock()
    model.noise.alpha_t = lambda t: torch.ones_like(t) * 0.5
    model._sigma_from_alphat = lambda alpha_t: torch.ones_like(alpha_t) * 0.5
    
    vocab_size = 50258
    def forward(x, sigma):
        batch_size, seq_len = x.shape
        # Return high probability for non-mask tokens
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 0] = 10.0  # High probability for token 0
        return logits
    model.forward = forward
    
    samples, mean_nfe = sampler.generate(
        model,
        num_samples=4,
        num_steps=10,
        eps=0.001,
        inject_bos=True
    )
    
    # All samples should complete (no masks remaining)
    assert (samples != model.mask_id).all(), "All masks should be replaced"
    # NFE should be recorded
    assert mean_nfe > 0


def test_nfe_all_complete_at_final_step():
    """Test NFE when all samples only complete at the noise removal step."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "sampling=default",
            "sampling.steps=5",
            "sampling.predictor=ddpm",
            "++sampling.nfe_metric=True",
        ])
        sampler = AbsorbingSampler(cfg)
    
    # Create a mock that keeps masks until final step
    model = MagicMock()
    model.num_tokens = 4
    model.device = torch.device("cpu")
    model.mask_id = 50256
    model.time_conditioning = False
    model.tokenizer = MagicMock()
    model.tokenizer.bos_token_id = 50257
    
    call_count = [0]
    
    def prior_sample(num_samples, num_tokens):
        return torch.full((num_samples, num_tokens), model.mask_id, dtype=torch.long)
    model.prior_sample = prior_sample
    
    model.noise = MagicMock()
    # Return alpha_t that causes very low denoising probability until final step
    def alpha_t(t):
        return t  # alpha_t = t, so prob_denoise = (alpha_s - alpha_t)/(1-alpha_t) is small
    model.noise.alpha_t = alpha_t
    model._sigma_from_alphat = lambda alpha_t: 1 - alpha_t
    
    vocab_size = 50258
    def forward(x, sigma):
        call_count[0] += 1
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 0] = 10.0
        return logits
    model.forward = forward
    
    samples, mean_nfe = sampler.generate(
        model,
        num_samples=2,
        num_steps=5,
        eps=0.001,
        inject_bos=True
    )
    
    # Should have called forward for each step + final noise removal
    assert call_count[0] == 6  # 5 steps + 1 noise removal


def test_self_speculative_sampler_nfe_tracking():
    """Test NFE tracking works with SelfSpeculativeSampler."""
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "sampling=self_speculative",
            "sampling.steps=5",
            "++sampling.nfe_metric=True",
        ])
        
        from discrete_diffusion.sampling.self_speculative import SelfSpeculativeSampler
        sampler = SelfSpeculativeSampler(cfg)
    
    # Create mock model compatible with SelfSpeculativeSampler
    model = MagicMock()
    model.num_tokens = 8
    model.device = torch.device("cpu")
    model.mask_id = 50256
    model.time_conditioning = False
    model.tokenizer = MagicMock()
    model.tokenizer.bos_token_id = 50257
    
    def prior_sample(num_samples, num_tokens):
        return torch.full((num_samples, num_tokens), model.mask_id, dtype=torch.long)
    model.prior_sample = prior_sample
    
    model.noise = MagicMock()
    model.noise.alpha_t = lambda t: torch.ones_like(t) * 0.5
    model._sigma_from_alphat = lambda alpha_t: torch.ones_like(alpha_t) * 0.5
    
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
    
    result = sampler.generate(
        model,
        num_samples=2,
        num_steps=5,
        eps=0.001,
        inject_bos=True
    )
    
    assert isinstance(result, tuple), "SelfSpeculativeSampler should return tuple with nfe_metric=True"
    samples, mean_nfe = result
    assert samples.shape == (2, model.num_tokens)
    assert isinstance(mean_nfe, float)
    assert mean_nfe > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

