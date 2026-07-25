# MRTF: Multimodal Reinforcement-Assisted Transformer Framework for Parkinson’s Disease Identification

This repository contains the official implementation of the proposed **Multimodal Reinforcement-Assisted Transformer Framework (MRTF)** for Parkinson’s disease (PD) identification using heterogeneous biomedical modalities.

MRTF integrates speech, structural MRI, and wearable sensor representations through a cross-cohort multimodal learning strategy. The framework combines:

- Modality-specific representation learning
- 512-dimensional latent-space alignment
- Cross-Attention Fusion Transformer (CAFT)
- Reinforcement-Assisted Self-Learning (RASL)
- Explainable AI Interactive Trust Layer (XAI-ITL)

The proposed framework is designed for cross-cohort multimodal representation learning when fully paired multimodal Parkinson’s datasets are unavailable.

---

## Framework Overview

MRTF consists of four major components:

1. **Modality-specific encoders**
   - Speech representation encoder using PC-GITA
   - MRI representation encoder using PPMI
   - Sensor representation encoder using Daphnet

2. **Latent Representation Alignment**
   - Modality-specific projection into a shared 512-dimensional embedding space

3. **Cross-Attention Fusion Transformer (CAFT)**
   - Learns statistical relationships among heterogeneous modality embeddings

4. **Explainability and Adaptive Optimization**
   - SHAP-based attribution
   - Counterfactual reasoning
   - Reinforcement-Assisted Self-Learning (RASL)

---

## Datasets

The framework uses publicly available datasets:

| Dataset | Modality | Usage |
|---|---|---|
| PC-GITA Parkinson Speech Corpus | Speech | Voice representation learning |
| Parkinson’s Progression Markers Initiative (PPMI) | MRI | Structural imaging representation learning |
| Daphnet Freezing-of-Gait Dataset | Wearable sensor | Parkinsonian motor representation learning |

Due to the absence of fully paired multimodal PD datasets, MRTF performs cross-cohort representation learning rather than patient-level multimodal fusion.

Raw clinical datasets are not included in this repository due to data access restrictions. Instructions for obtaining the datasets are provided in:
