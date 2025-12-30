#!/bin/bash
# Debug script for all sampling configurations
# Runs with num_samples=8, saves trajectories and text examples
# Simplified: fixed diffusion_temperature=1.1, num_steps=256

set -e

# Change to project root (parent of examples/)
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONPATH=src

# Common settings for debug mode
DEBUG_ARGS="num_samples=8 devices=8 device=cuda max_length=512 num_steps=256 sampling.sampler.diffusion_temperature=1.1"

echo "=============================================="
echo "Running DEBUG experiments (num_samples=8)"
echo "=============================================="

# =============================================
# 1. MDLM
# =============================================
echo ""
echo ">>> 1. MDLM Debug Experiments"
echo ""

python -m discrete_diffusion.evaluations.generate_samples \
    'experiment=[sampling/mdlm,sampling/debug]' \
    ${DEBUG_ARGS} \
    base_samples_path=example_samples/mdlm

# =============================================
# 2. GStar - Grid search over remasker_temperature
# =============================================
echo ""
echo ">>> 2. GStar Debug Experiments"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    'experiment=[sampling/gstar,sampling/debug]' \
    ${DEBUG_ARGS} \
    base_samples_path=example_samples/gstar \
    sampling.sampler.remasker_temperature=0,1

# =============================================
# 3. GStar + Finetune Backbone
# =============================================
echo ""
echo ">>> 3. GStar + Finetune Backbone Debug Experiments"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    'experiment=[sampling/gstar,sampling/debug]' \
    ${DEBUG_ARGS} \
    checkpoint_path=outputs/owt/gstar_len512_finetune_backbone/dummy_checkpoints/checkpoints/best.ckpt \
    base_samples_path=example_samples/gstar_finetune_backbone \
    sampling.sampler.remasker_temperature=0,1

# =============================================
# 4. GStar + Finetune Backbone + Time Conditioning
# =============================================
echo ""
echo ">>> 4. GStar + Finetune Backbone + Time Conditioning Debug Experiments"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    'experiment=[sampling/gstar,sampling/debug]' \
    ${DEBUG_ARGS} \
    checkpoint_path=outputs/owt/gstar_len512_time_conditioning_finetune_backbone/dummy_checkpoints/checkpoints/best.ckpt \
    base_samples_path=example_samples/gstar_time_cond_finetune_backbone \
    sampling.sampler.remasker_temperature=0,1

# =============================================
# 5. StarShape (MDLM backbone with StarShape sampler)
#    Two regimes: default and plato
# =============================================
echo ""
echo ">>> 5. StarShape Debug Experiments"
echo ""

# Regime A: remasker_schedule=default, t_on=0.2, t_off=0
python -m discrete_diffusion.evaluations.generate_samples \
    'experiment=[sampling/starshape,sampling/debug]' \
    ${DEBUG_ARGS} \
    base_samples_path=example_samples/starshape \
    sampling.sampler.remasker_schedule=default \
    sampling.sampler.t_on=0.2 \
    sampling.sampler.t_off=0

# Regime B: remasker_schedule=plato, t_on=0.55, t_off=0.05
python -m discrete_diffusion.evaluations.generate_samples \
    'experiment=[sampling/starshape,sampling/debug]' \
    ${DEBUG_ARGS} \
    base_samples_path=example_samples/starshape \
    sampling.sampler.remasker_schedule=plato \
    sampling.sampler.t_on=0.55 \
    sampling.sampler.t_off=0.05

echo ""
echo "=============================================="
echo "All DEBUG experiments completed!"
echo "=============================================="

