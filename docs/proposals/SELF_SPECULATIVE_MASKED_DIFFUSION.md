# Self-Speculative Masked Diffusion Implementation Proposal

**Paper**: [Self-Speculative Masked Diffusions](https://arxiv.org/pdf/2510.03929v1)  
**Authors**: Campbell, De Bortoli, Shi, Doucet (Google DeepMind)  
**Task**: Text Generation on OpenWebText (OWT)

---

## Overview

This proposal outlines the implementation of Self-Speculative Masked Diffusions for text generation. The method achieves ~2× reduction in Number of Function Evaluations (NFE) by enabling non-factorized predictions over masked positions through a hybrid non-causal/causal architecture with speculative sampling.

**Our Simplified Approach**: We use 12 non-causal layers + 1 causal layer (13 total) instead of the paper's 11+1=12. This small overhead allows us to freeze the entire MDLM backbone and train only the causal layer + target head, significantly reducing training cost.

---

## Step 0: Scope Definition

**Focus**: Text generation on OpenWebText (OWT) only.

**Out of Scope**:
- Text8 dataset (omitted for simplicity)
- Protein sequence generation (UniRef50)
- Other discrete diffusion variants

---

## Step 1: MDLM + AR Layer Architecture

### Description

Implement a hybrid transformer architecture consisting of:
1. **Non-causal backbone**: Full pretrained MDLM transformer with bidirectional attention (N layers, frozen)
2. **Causal head layer**: Single transformer block with causal (autoregressive) attention mask (trained)
3. **Two distinct output heads**: Draft head (frozen) and Target head (trained)

### Architecture Comparison

| Aspect | Paper (OWT) | Our Approach |
|--------|-------------|--------------|
| Non-causal layers | 11 (frozen in our interp.) | **12** (frozen, full MDLM) |
| Causal layer | 1 (layer 12) | **1** (layer 13, added) |
| Total layers | 12 | **13** |
| Draft head | Shared or separate | **Pretrained MDLM head** (frozen) |
| Target head | Shared or separate | **New head** (trained) |
| Training | Full fine-tune (1M steps) | **Only causal layer + target head (50K steps)** |

### Architecture Diagram

```
                        FROZEN (Pretrained MDLM)
                    ┌─────────────────────────────────────┐
Input → Embedding → │ Non-causal Layer 1                  │
                    │ Non-causal Layer 2                  │
                    │ ...                                 │
                    │ Non-causal Layer 12                 │ → Hidden H₁₂
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           Draft Head (frozen)              Causal Layer 13 (TRAINED)
           (pretrained MDLM head)                   │
                    │                               ▼
                    ▼                            Hidden H₁₃
             DRAFT LOGITS                           │
                                                    ▼
                                          Target Head (TRAINED)
                                          (new, randomly init)
                                                    │
                                                    ▼
                                             TARGET LOGITS
```

### Key Design Decisions

1. **Two distinct heads**: 
   - Draft head = pretrained MDLM output projection (frozen)
   - Target head = new output projection (trained with causal layer)
   - This allows draft quality to match original MDLM

2. **Permutation-informed causal masking**: The causal layer attends based on the generation order σ, not left-to-right

3. **Double position encoding**: Encode both current position and next-in-order position (for RoPE: split channels in half)

4. **Information flow**: Causal layer receives hidden states from non-causal layers plus additional revealed token information

### Input/Output Specification

**Inputs**:
- `x_masked`: Partially masked token sequence [B, L]
- `sigma`: Permutation order for generation [B, L]
- `t`: Timestep / mask ratio

**Outputs**:
- `draft_logits`: From draft head applied to H₁₂ [B, L, V]
- `target_logits`: From target head applied to H₁₃ [B, L, V]

### Success Criteria

#### Unit Tests: `tests/test_models/test_self_spec_architecture.py`
- [ ] `test_causal_mask_generation`: Verify causal attention mask correctly follows permutation order σ
- [ ] `test_draft_target_split`: Draft uses frozen head, target uses trained head
- [ ] `test_forward_pass`: Model produces valid logits for both draft and target outputs
- [ ] `test_gradient_flow`: Gradients only flow through causal layer and target head
- [ ] `test_double_position_encoding`: Both current and next-in-order positions encoded correctly

#### Unit Tests: `tests/test_models/test_self_spec_loading.py`
- [ ] `test_model_loading`: Can load pretrained MDLM weights into backbone + draft head
- [ ] `test_weight_freezing`: Backbone and draft head parameters frozen correctly
- [ ] `test_target_head_init`: Target head initialized randomly

#### Integration Tests: `tests/test_integration/test_self_spec_model.py`
- [ ] `test_inference_mode`: Model runs correctly in eval mode
- [ ] `test_batch_processing`: Model handles batched inputs correctly

---

## Step 2: Training Scheme

### Description

Implement the training procedure for the Self-Speculative MDLM.

**Our simplified approach**:
1. **FREEZE** the entire pretrained MDLM (all 12 non-causal layers + draft head)
2. **Train only** the new causal layer (layer 13) + target head

This is more efficient than the paper's full fine-tuning approach.

### Training Objective

Standard masked language modeling loss on the **target head** outputs:

$$\mathcal{L} = \mathbb{E}_{\sigma, t, x} \left[ \sum_{d: x_t^d = M} -\log p_\theta^{\text{target}}(x^{\sigma(d)} | x^{\sigma(1:d-1)}) \right]$$

### Permutation (σ) Sampling

**From paper**: *"Let σ be a permutation of the numbers {1,...,D}... This permutation is usually picked uniformly at random."*

The permutation σ is:
- Sampled **uniformly at random** over all possible orderings
- Applied to the **currently masked positions** only
- **NOT** conditioned on which tokens are masked - it's simply a random ordering
- During training: random σ sampled each forward pass
- During inference: random σ sampled for remaining masked positions at each step

### Training Configuration

#### Paper's OWT Configuration (Appendix E.2)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | 11 NC + 1 C = 12 layers | Full fine-tuning |
| Training steps | **1,000,000** | Very long |
| Batch size | 256 | |
| Sequence length | 1024 | |
| Optimizer | AdamW | β₁=0.9, β₂=0.999 |
| Learning rate | 1.5×10⁻⁴ | Cosine decay |
| Warmup steps | 10,000 | |
| Weight decay | 0.01 | |
| Gradient clipping | 1.0 | Norm clipping |
| Dropout | 0.1 | |

#### Our Simplified Configuration (OpenWebText)

Since we train only 1 causal layer + target head (~10% of params), we use **50K steps**.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Frozen layers | 1 to 12 + draft head | Pretrained MDLM |
| Trainable | Causal layer 13 + target head | ~10% of params |
| Training iterations | **50,000** | Only 1 layer to train |
| Batch size | 256 | Match paper |
| Sequence length | 1024 | Match paper |
| Optimizer | AdamW | β₁=0.9, β₂=0.999 |
| Learning rate | 1.5×10⁻⁴ | Peak LR |
| LR schedule | Cosine decay to 0 | Final LR < 1e-6 × peak |
| Warmup steps | **500** | 1% of training (linear warmup) |
| Weight decay | 0.01 | Match paper |
| Gradient clipping | 1.0 | Match paper |
| Dropout | 0.1 | Match paper |
| Checkpoint interval | 10,000 | 5 checkpoints total |
| Validation interval | 5,000 | 10 validations total |

#### Learning Rate Schedule Details

We use **linear warmup + cosine decay**:

$$\eta(t) = \begin{cases} 
\eta_{max} \cdot \frac{t}{t_{warmup}} & \text{if } t < t_{warmup} \\
\eta_{max} \cdot \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{t - t_{warmup}}{T - t_{warmup}}\right)\right) & \text{otherwise}
\end{cases}$$

At end of training (t = T = 50,000):
- cos(π) = -1
- η(T) = η_max × 0.5 × (1 - 1) = **0**

This ensures the final learning rate is essentially 0, satisfying the < 1e-6 requirement.

| Step | LR Coefficient | Actual LR |
|------|---------------|-----------|
| 0 | 0 | 0 |
| 250 | 0.5 | 7.5×10⁻⁵ |
| 500 (end warmup) | 1.0 | 1.5×10⁻⁴ |
| 25,000 (mid) | 0.5 | 7.5×10⁻⁵ |
| 50,000 (end) | 0 | 0 |

### Success Criteria

#### Unit Tests: `tests/test_training/test_self_spec_training.py`
- [ ] `test_frozen_backbone`: Verify backbone parameters have `requires_grad=False`
- [ ] `test_frozen_draft_head`: Verify draft head parameters have `requires_grad=False`
- [ ] `test_trainable_causal_layer`: Verify causal layer parameters have `requires_grad=True`
- [ ] `test_trainable_target_head`: Verify target head parameters have `requires_grad=True`
- [ ] `test_loss_computation`: Loss computed correctly on masked positions only using target logits

#### Unit Tests: `tests/test_training/test_permutation.py`
- [ ] `test_permutation_uniform_sampling`: Permutations sampled uniformly at random
- [ ] `test_permutation_valid`: All permutations are valid (bijective)
- [ ] `test_permutation_batch`: Batch of independent permutations generated correctly

#### Integration Tests: `tests/test_integration/test_self_spec_training.py`
- [ ] `test_training_step`: Single training step executes without errors
- [ ] `test_checkpoint_save_load`: Can save and resume training from checkpoint
- [ ] `test_loss_decreases`: Target head loss decreases over training iterations
- [ ] `test_lr_schedule`: LR follows warmup + cosine decay, ends at ~0

#### E2E Test: `tests/test_e2e/test_self_spec_training.py`

End-to-end training smoke test that runs a short training loop to verify everything works together.

```python
def test_training_e2e():
    """
    Run 100 training steps + validation to verify end-to-end training works.
    
    Config:
    - Training steps: 100
    - Validation every: 50 steps
    - Warmup steps: 10
    - Small batch size: 4
    - Short sequence length: 128
    """
```

**Test Checks:**
- [ ] Model initializes correctly (frozen backbone + trainable causal layer)
- [ ] Training loop runs 100 steps without errors
- [ ] Validation runs at steps 50 and 100
- [ ] Loss values are finite (no NaN/Inf)
- [ ] Training loss decreases from step 0 to step 100
- [ ] Target loss < Draft loss after training
- [ ] Gradients only update causal layer + target head (frozen params unchanged)
- [ ] Checkpoint saves and loads correctly
- [ ] LR schedule follows expected warmup + decay pattern

#### Training Metrics to Track
- [ ] Training loss (target head)
- [ ] Validation loss (both draft and target)
- [ ] Learning rate
- [ ] Verify: target loss < draft loss (target should be strictly better)

---

## Step 3: Self-Speculative Decoding Scheme

### Description

Implement the speculative sampling algorithm for masked diffusion inference:

1. **Draft phase**: Generate K draft tokens using draft head (from frozen MDLM)
2. **Verify phase**: Compute target probabilities for all K tokens in parallel using target head
3. **Accept/Reject**: Use speculative sampling to accept a prefix of valid tokens

### Speculation Length (K) and Δτ Settings

**From paper (Appendix E.2)**: The paper provides specific settings for the trade-off between quality and NFE:

| Draft/Verify Steps per NC Pass | Cosine Window Δτ | Approx NFE |
|-------------------------------|------------------|------------|
| 1 | 0.002 | High (~500) |
| 2 | 0.005 | ~250 |
| 3 | 0.01 | ~170 |
| 4 | 0.02 | ~128 |
| 8 | 0.04 | ~64 |
| 10 | 0.083 | ~28 |

**K selection**: 
- K = number of draft tokens to generate per speculative step
- Higher K = more aggressive speculation, fewer forward passes, but potentially more rejections
- For NFE=64: K ≈ 8 draft/verify steps
- For NFE=28: K ≈ 10 draft/verify steps

### Algorithm

```
Input: x_t (partially masked sequence), K (speculation length)
Output: x_{t'} (sequence with more tokens revealed)

1. Identify remaining masked positions M = {i : x_t[i] = MASK}
2. Sample permutation σ uniformly at random over M
3. For i = 1 to K:
   - Compute draft logits from frozen MDLM (draft head)
   - Sample draft token: d_i ~ p_draft(· | x_t, d_{1:i-1})
4. Single forward pass through causal layer:
   - Compute target logits for all K positions (target head)
   - p_target(d_1), p_target(d_2 | d_1), ..., p_target(d_K | d_{1:K-1})
5. For i = 1 to K:
   - r = p_target(d_i | d_{1:i-1}) / p_draft(d_i | d_{1:i-1})
   - Accept d_i with probability min(1, r)
   - If rejected: sample correction from residual distribution, break
6. Return updated sequence with accepted tokens revealed
```

### Key Implementation Details

1. **Draft from frozen MDLM**: Draft logits come from the pretrained draft head (frozen), ensuring high-quality drafts

2. **Parallel verification**: All K target probabilities computed in single forward pass through causal layer + target head

3. **Residual sampling**: When rejection occurs, sample from `max(0, p_target - p_draft)` normalized

4. **NFE counting**: 
   - Each non-causal forward pass = 1 NFE
   - Draft generation reuses cached hidden states from non-causal pass
   - Causal layer verification = additional compute but same NFE

### Success Criteria

#### Unit Tests: `tests/test_sampling/test_self_spec_draft.py`
- [ ] `test_draft_from_frozen_head`: Draft tokens sampled from frozen draft head
- [ ] `test_draft_autoregressive`: Draft tokens generated sequentially using cached states

#### Unit Tests: `tests/test_sampling/test_self_spec_verify.py`
- [ ] `test_parallel_verification`: Target probabilities computed in single causal layer pass
- [ ] `test_acceptance_probability`: Accept/reject computed correctly per speculative sampling
- [ ] `test_residual_distribution`: Correction samples drawn from valid residual

#### Unit Tests: `tests/test_sampling/test_self_spec_correctness.py`
- [ ] `test_distribution_correctness`: Final tokens distributed according to target (statistical test)
- [ ] `test_nfe_counting`: NFE counted accurately during generation

#### Integration Tests: `tests/test_integration/test_self_spec_sampling.py`
- [ ] `test_full_generation`: Generate complete sequence from fully masked input
- [ ] `test_deterministic_seed`: Same seed produces same output
- [ ] `test_variable_length`: Works with different sequence lengths
- [ ] `test_variable_k`: Works with different speculation lengths K

---

## Step 4: Training Self-Speculative MDLM

### Description

Train the complete Self-Speculative MDLM model on OpenWebText.

### Prerequisites
- Pretrained MDLM on OWT (12 layers)

### Training Pipeline
```
1. Load pretrained MDLM checkpoint (12 layers + head)
2. Freeze all 12 layers + head (this becomes draft head)
3. Add new causal layer 13 (randomly initialized)
4. Add new target head (randomly initialized)
5. Train for 50K iterations on OWT
6. Save final checkpoint
```

### Checkpoints
| Checkpoint | Description |
|------------|-------------|
| `owt_mdlm_pretrained.pt` | Original MDLM (for draft) |
| `owt_self_spec_mdlm_final.pt` | Fully trained model |
| `owt_self_spec_mdlm_iter_*.pt` | Intermediate checkpoints (every 10K) |

### Success Criteria

#### Training Completion
- [ ] Training completes without crashes
- [ ] Final target loss < draft loss (verifies target is strictly better than draft)

#### Validation During Training
- [ ] Track validation loss every 5K steps (both draft and target)
- [ ] Verify target loss consistently lower than draft loss

---

## Step 5: Evaluation

### Description

Evaluate the trained Self-Speculative MDLM following paper methodology.

### Metrics (from paper)

The paper reports the following metrics:

#### 1. GPT-2 NLL (Negative Log-Likelihood)

This is the primary quality metric, equivalent to generative perplexity evaluated by GPT-2:

$$\text{GPT-2 NLL} = -\frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L} \log p_{\text{GPT2}}(x_j^{(i)} | x_{<j}^{(i)})$$

**Note**: Lower is better. This measures how "realistic" generated samples are according to GPT-2.

#### 2. Entropy

Measures the diversity/uncertainty in generated samples:

$$\text{Entropy} = -\sum_{x} p(x) \log p(x)$$

Computed over token distributions or n-gram distributions of generated samples.

#### 3. Number of Function Evaluations (NFE)

$$\text{NFE} = \text{Number of neural network forward passes to generate one sample}$$

This is the efficiency metric. Lower NFE = faster generation.

**Note**: Diversity metrics like Self-BLEU and Distinct-n are **NOT** reported in the paper and should be omitted.

### Evaluation Protocol (OpenWebText)

| Configuration | NFE | Target GPT-2 NLL | Notes |
|--------------|-----|------------------|-------|
| Baseline MDLM | 512 | Reference | Quality baseline |
| Self-Spec | 28 | Comparable | ~18× fewer NFE (paper result) |
| Self-Spec | 64 | ≤ Baseline | ~8× fewer NFE |
| Baseline MDLM | 28 | >> Self-Spec | Much worse (paper shows gibberish) |

### Sample Generation Protocol

- Generate 1000 samples of length 1024
- Evaluate at NFE ∈ {28, 64, 128, 256, 512}

### Success Criteria

#### Unit Tests: `tests/test_evaluation/test_metrics.py`
- [ ] `test_gpt2_nll_computation`: GPT-2 NLL computed correctly
- [ ] `test_entropy_computation`: Entropy computed correctly
- [ ] `test_nfe_counting`: NFE counted accurately during generation

#### Integration Tests: `tests/test_integration/test_evaluation.py`
- [ ] `test_full_evaluation_pipeline`: Complete evaluation runs end-to-end
- [ ] `test_baseline_comparison`: Can evaluate both baseline and self-spec models

#### Target Metrics (OpenWebText)

| Metric | Self-Spec @ NFE=28 | Baseline @ NFE=28 | Success Criterion |
|--------|-------------------|-------------------|-------------------|
| GPT-2 NLL | ~3.5-4.0 | >5.0 | **Significantly better** |
| Sample quality | Coherent text | Gibberish | Qualitative |

| Metric | Self-Spec @ NFE=64 | Baseline @ NFE=256 | Success Criterion |
|--------|-------------------|-------------------|-------------------|
| GPT-2 NLL | X | Y | X ≈ Y (within 10%) |
| NFE | 64 | 256 | **~2× reduction** |

---

## Summary: Key Deliverables

### Code Deliverables
1. [ ] `SelfSpecMDLM` model class with hybrid architecture (12 NC + 1 C + 2 heads)
2. [ ] Training script with frozen backbone + draft head
3. [ ] Self-speculative sampling implementation
4. [ ] Evaluation scripts for GPT-2 NLL, Entropy, NFE

### Test Files Structure
```
tests/
├── test_models/
│   ├── test_self_spec_architecture.py
│   └── test_self_spec_loading.py
├── test_training/
│   ├── test_self_spec_training.py
│   └── test_permutation.py
├── test_sampling/
│   ├── test_self_spec_draft.py
│   ├── test_self_spec_verify.py
│   └── test_self_spec_correctness.py
├── test_evaluation/
│   └── test_metrics.py
├── test_integration/
│   ├── test_self_spec_model.py
│   ├── test_self_spec_training.py
│   ├── test_self_spec_sampling.py
│   └── test_evaluation.py
└── test_e2e/
    └── test_self_spec_training.py      # 100-step training smoke test
```

### Checkpoints
1. [ ] OWT: Pretrained MDLM (12 layers)
2. [ ] OWT: Trained Self-Speculative MDLM (12 NC + 1 C + 2 heads)

### Documentation
1. [ ] Training instructions
2. [ ] Evaluation instructions
3. [ ] Results reproduction guide

### Final Success Metric

Achieve **~2× reduction in NFE** at equivalent GPT-2 NLL compared to baseline MDLM on OpenWebText, matching paper results.

---

## Appendix: Differences from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Architecture | 11 NC + 1 C = 12 total | 12 NC + 1 C = 13 total |
| Output heads | 2 (likely shared embedding) | 2 distinct (draft frozen, target trained) |
| Training | Full fine-tuning (1M steps) | Freeze backbone, train only C layer + target head |
| Training steps | 1M | **50K** |
| Warmup steps | 10K | **500** (1% of training) |
| LR schedule | Cosine decay | Cosine decay to 0 (final LR < 1e-6) |
| Resource usage | High (full model training) | **Low** (only ~10% params trained) |

---

## References

1. Campbell, A., De Bortoli, V., Shi, J., & Doucet, A. (2024). Self-Speculative Masked Diffusions. arXiv:2510.03929
2. Sahoo et al. (2024). Simple and Effective Masked Diffusion Language Models (MDLM)
3. Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding
