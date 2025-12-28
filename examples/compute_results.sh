#!/bin/bash
# Full experiments script for all sampling configurations
# Runs with num_samples=5000, computes gen ppl and diversity metrics
# Uses --multirun for grid search over all parameters
# Metrics are computed immediately after each sampling step

set -e

# Change to project root (parent of examples/)
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONPATH=src

# Common settings
COMMON_ARGS="devices=8 device=cuda max_length=512"
METRICS_ARGS="devices=8 device=cuda max_length=512"

# Function to compute metrics for all .pt files in a directory
compute_metrics() {
    local samples_dir=$1
    echo ""
    echo "  >> Computing metrics for samples in: ${samples_dir}"
    
    for pt_file in ${samples_dir}/*.pt; do
        if [ -f "$pt_file" ]; then
            # Check if metrics already exist
            metrics_file="${pt_file/samples\//metrics\/}"
            metrics_file="${metrics_file%.pt}.json"
            if [ -f "$metrics_file" ]; then
                echo "    Skipping (metrics exist): $pt_file"
            else
                echo "    Processing: $pt_file"
                python -m discrete_diffusion.evaluations.generative_ppl \
                    samples_path="$pt_file" \
                    ${METRICS_ARGS}
            fi
        fi
    done
}

echo "=============================================="
echo "Running FULL experiments (num_samples=5000)"
echo "=============================================="

# =============================================
# 1. MDLM - Sampling + Metrics
# =============================================
echo ""
echo ">>> 1. MDLM Sampling"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    experiment=sampling/mdlm \
    ${COMMON_ARGS} \
    base_samples_path=samples/mdlm \
    sampling.use_float64=true \
    sampling.p_nucleus=0.9 \
    sampling.sampler.diffusion_temperature=0.9,1.0,1.1,1.2 \
    num_steps=128,256,512

echo ""
echo ">>> 1. MDLM Metrics"
compute_metrics "samples/mdlm"

# =============================================
# 2. GStar - Sampling + Metrics
# =============================================
echo ""
echo ">>> 2. GStar Sampling"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    experiment=sampling/gstar \
    ${COMMON_ARGS} \
    base_samples_path=samples/gstar \
    sampling.use_float64=true \
    sampling.p_nucleus=0.9 \
    sampling.sampler.diffusion_temperature=0.9,1.0,1.1,1.2 \
    sampling.sampler.remasker_temperature=0,1 \
    num_steps=128,256,512

echo ""
echo ">>> 2. GStar Metrics"
compute_metrics "samples/gstar"

# =============================================
# 3. GStar + Finetune Backbone - Sampling + Metrics
# =============================================
echo ""
echo ">>> 3. GStar + Finetune Backbone Sampling"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    experiment=sampling/gstar \
    ${COMMON_ARGS} \
    checkpoint_path=outputs/owt/gstar_len512_finetune_backbone/dummy_checkpoints/checkpoints/best.ckpt \
    base_samples_path=samples/gstar_finetune_backbone \
    sampling.use_float64=true \
    sampling.p_nucleus=0.9 \
    sampling.sampler.diffusion_temperature=0.9,1.0,1.1,1.2 \
    sampling.sampler.remasker_temperature=0,1 \
    num_steps=128,256,512

echo ""
echo ">>> 3. GStar + Finetune Backbone Metrics"
compute_metrics "samples/gstar_finetune_backbone"

# =============================================
# 4. GStar + Finetune Backbone + Time Conditioning - Sampling + Metrics
# =============================================
echo ""
echo ">>> 4. GStar + Finetune Backbone + Time Conditioning Sampling"
echo ""

python -m discrete_diffusion.evaluations.generate_samples --multirun \
    experiment=sampling/gstar \
    ${COMMON_ARGS} \
    checkpoint_path=outputs/owt/gstar_len512_time_conditioning_finetune_backbone/dummy_checkpoints/checkpoints/best.ckpt \
    base_samples_path=samples/gstar_time_cond_finetune_backbone \
    sampling.use_float64=true \
    sampling.p_nucleus=0.9 \
    sampling.sampler.diffusion_temperature=0.9,1.0,1.1,1.2 \
    sampling.sampler.remasker_temperature=0,1 \
    num_steps=128,256,512

echo ""
echo ">>> 4. GStar + Finetune Backbone + Time Conditioning Metrics"
compute_metrics "samples/gstar_time_cond_finetune_backbone"

# =============================================
# 5. StarShape (MDLM backbone with StarShape sampler)
#    Two regimes: default and plato
# =============================================
echo ""
echo ">>> 5. StarShape Sampling (Regime A: default schedule)"
echo ""

# Regime A: remasker_schedule=default, t_on=0.2, t_off=0
python -m discrete_diffusion.evaluations.generate_samples --multirun \
    experiment=sampling/starshape \
    ${COMMON_ARGS} \
    base_samples_path=samples/starshape \
    sampling.sampler.remasker_schedule=default \
    sampling.sampler.t_on=0.2 \
    sampling.sampler.t_off=0 \
    sampling.use_float64=true \
    sampling.p_nucleus=0.9 \
    sampling.sampler.diffusion_temperature=0.9,1.0,1.1,1.2 \
    num_steps=128,256,512

echo ""
echo ">>> 5. StarShape Sampling (Regime B: plato schedule)"
echo ""

# Regime B: remasker_schedule=plato, t_on=0.55, t_off=0.05
python -m discrete_diffusion.evaluations.generate_samples --multirun \
    experiment=sampling/starshape \
    ${COMMON_ARGS} \
    base_samples_path=samples/starshape \
    sampling.sampler.remasker_schedule=plato \
    sampling.sampler.t_on=0.55 \
    sampling.sampler.t_off=0.05 \
    sampling.use_float64=true \
    sampling.p_nucleus=0.9 \
    sampling.sampler.diffusion_temperature=0.9,1.0,1.1,1.2 \
    num_steps=128,256,512

echo ""
echo ">>> 5. StarShape Metrics"
compute_metrics "samples/starshape"

echo ""
echo "=============================================="
echo "All experiments and metrics computation completed!"
echo "Metrics saved in 'metrics/' directory"
echo "=============================================="

