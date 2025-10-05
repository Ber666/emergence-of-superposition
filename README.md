# Emergence of Superposition

Official implementation for **[Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](https://arxiv.org/pdf/2509.23365)**.

## Overview

This work investigates the training dynamics of Chain of Continuous Thought ([Coconut](https://arxiv.org/abs/2412.06769)). This repository includes:

- **ProsQA Dataset**: Synthetic graph search dataset requiring multi-hop reasoning. Different from previous work ([Coconut](https://github.com/facebookresearch/coconut), [Reasoning-by-Superposition](https://github.com/Ber666/reasoning-by-superposition)), the datasets are shuffled to better align with the theory.

- **Coconut and Coconut-BFS Training**: Updated code for training both standard Coconut and Coconut-BFS. The BFS loss function is described in Appendix E.2 of the paper.

- **Attention Analysis Tools**: Code to analyze attention patterns and validate the training dynamics (Theorem 1 in the paper).

## Setup

Clone and setup environment:
```bash
git clone https://github.com/Ber666/emergence-of-superposition.git
cd emergence-of-superposition

conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

Login to WandB for experiment tracking:
```bash
wandb login
```

## Quick Start

### Training

**With BFS Loss**:
```bash
bash launch_exp.sh args/coconut_bfs_loss.yaml
```

**Standard Coconut**:
```bash
bash launch_exp.sh args/coconut_standard.yaml
```

### Evaluation

```bash
bash launch_exp.sh args/coconut_bfs_loss_eval.yaml
```

**Note**: Update `load_model_path` in the eval config to point to your trained checkpoint.

## Visualization

### Attention Evolution Analysis

Analyze how the model learns to attend to frontier edges:

```bash
python plot_attention_across_epochs.py \
    --checkpoint_dir ckpts/coconut-bfs-loss \
    --epochs 10 20 30 40 50 60 70 \
    --steps 0 1 \
    --output attention_evolution.png \
    --test_file data/prosqa_test_graph_4_coconut_shuffled_with_bfs.json
```

This generates:
- `attention_evolution.png` - Visualization of attention patterns
- `attention_evolution_raw_data.json` - Raw analysis data

### Create Publication Plots

Convert raw data to PDF:

```bash
python plot_from_raw_data.py \
    attention_evolution_raw_data.json \
    --output attention_evolution.pdf
```

## Configuration

Key training parameters in `args/*.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `bfs-paper-loss` | Use BFS loss function | True/False |
| `num_epochs` | Total training epochs | 350 |
| `first_stage_epochs` | Epochs for stage 0 (no thoughts) | 100 |
| `epochs_per_stage` | Epochs per reasoning depth | 50 |
| `max_latent_stage` | Maximum reasoning steps | 4 |
| `batch_size_training` | Batch size per GPU | 256 |
| `lr` | Learning rate | 1e-4 |
| `shuffle_nodes` | Randomize node indices | True |

## Hardware Requirements

- **Training**: 2 GPUs (configured for FSDP distributed training)
- **Evaluation**: 2 GPUs
- **Tested on**: A100 GPUs

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhu2025emergence,
  title={Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought},
  author={Zhu, Hanlin and Hao, Shibo and Hu, Zhiting and Jiao, Jiantao and Russell, Stuart and Tian, Yuandong},
  journal={arXiv preprint arXiv:2509.23365},
  year={2025}
}
```

## License

MIT License (see [LICENSE](LICENSE))
