# BAPM: Bidirectional Attention-based Pose Matching

Official PyTorch implementation of the CVPR 2026 paper:

> **BAPM: Bidirectional Attention-based Pose Matching**
> 
> [Paper](#) | [Project Page](#) | [ArXiv](#)

## Introduction

BAPM is a novel framework for pose matching that leverages bidirectional attention mechanisms to capture long-range dependencies between feature representations. Our method achieves state-of-the-art performance on standard benchmarks.

![BAPM Framework](assets/framework.png)

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3

```bash
# Clone the repository
git clone https://github.com/leavlong/BAPM.git
cd BAPM

# Create a conda environment
conda create -n bapm python=3.8
conda activate bapm

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

Please refer to [DATASET.md](docs/DATASET.md) for dataset download and preparation instructions.

The datasets should be organized as:
```
data/
├── dataset_name/
│   ├── train/
│   ├── val/
│   └── test/
```

## Training

```bash
# Train on a single GPU
python train.py --config configs/bapm_default.yaml

# Train with multiple GPUs (e.g., 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --config configs/bapm_default.yaml \
    --distributed
```

## Evaluation

```bash
python test.py --config configs/bapm_default.yaml \
    --checkpoint checkpoints/bapm_best.pth
```

## Model Zoo

| Model | Dataset | Metric | Download |
|-------|---------|--------|----------|
| BAPM  | —       | —      | [model](#) |

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{bapm2026,
  title={BAPM: Bidirectional Attention-based Pose Matching},
  author={},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

We thank the open-source community for their contributions to the tools and libraries used in this work.