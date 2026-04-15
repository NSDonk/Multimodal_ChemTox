# Multimodal_ChemTox

## Learning Multimodal Chemical Representations for Toxicity Prediction

Multimodal deep learning framework for chemical toxicity prediction combining 
Vision Transformer (ViT) image encoders with molecular descriptor MLPs. Developed 
as a project for Generative AI for Biomedicine course at Carnegie Mellon University.

## Overview

Traditional QSAR models rely on hand-engineered descriptors and struggle to 
generalize across chemical space. This project develops a unified ViT + descriptor 
fusion architecture with two key additions:

- **Deterministic molecular image augmentation** — rotations and Gaussian noise 
  applied to 2D molecular structure images to expand training data
- **Masked-descriptor reconstruction pretraining objective** — encourages the 
  descriptor encoder to learn chemically meaningful representations before 
  task-specific finetuning

The model is pretrained on ToxCast (8,579 molecules) and finetuned on Tox21 
(7,823 molecules), with separate prediction heads for each dataset.

## Architecture

- **Image branch**: ViT encoder (8/16 patch, 12 transformer layers, D=768) 
  processing 224×224 RGB molecular images
- **Descriptor branch**: MLP with LayerNorm and Dropout processing 2048-bit 
  molecular fingerprints (D=256)
- **Fusion**: Concatenation layer (D=512) feeding into task-specific linear heads
- **Pretraining**: Masked MSE reconstruction objective on descriptor encoder

## Experiments

Ablation study varying ViT trainability (frozen / partial / full), descriptor and 
fusion layer inclusion, and ToxCast pretraining before Tox21 finetuning. Key 
conditions tested: ViT-only, Descriptor-only, Multimodal (Frozen), Multimodal 
(Partial Unfreeze), Multimodal (Full).

## Results

- Multimodal model outperforms unimodal baselines on Tox21
- ToxCast pretraining improves transfer stability to Tox21
- Ablation study shows partial ViT unfreezing with descriptor fusion achieves 
  best generalization
- UMAP analysis confirms ToxCast and Tox21 occupy overlapping chemical space, 
  supporting cross-dataset generalization
- Observed Tox21 overfitting attributed to ToxCast label sparsity rather than 
  structural domain mismatch

## Dependencies

- PyTorch
- timm (ViT implementations)
- RDKit (molecular image generation, fingerprints)
- scikit-learn, numpy, matplotlib

## Reference

Hong & Kwon (2025). Multimodal deep learning for chemical toxicity prediction 
and management. *Scientific Reports* 15, 19491.
