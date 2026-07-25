# MRTF Dataset Description and Preparation

This directory contains dataset documentation, metadata files, and preparation scripts required to reproduce the experiments reported in the MRTF manuscript.

The MRTF framework utilizes three independent public Parkinson's disease datasets:

1. PC-GITA Parkinson Speech Corpus
2. Parkinson's Progression Markers Initiative (PPMI)
3. Daphnet Freezing-of-Gait Dataset


## Important Notice

The original datasets are not redistributed in this repository due to their respective licenses and access restrictions.

Users must obtain the datasets from the official repositories and follow the corresponding usage agreements.


# Dataset Overview


## 1. PC-GITA Parkinson Speech Corpus

Role in MRTF:

- Modality:
  Speech

- Purpose:
  Voice representation learning

- Subjects:
  100 participants

  - 50 Parkinson's disease subjects
  - 50 healthy controls


- Used information:

  - sustained phonation
  - reading tasks
  - diadochokinetic articulation exercises


- Sampling frequency:

  44.1 kHz


- Processing:

  - MFCC extraction
  - jitter
  - shimmer
  - HNR
  - spectrogram generation



## 2. Parkinson's Progression Markers Initiative (PPMI)

Role in MRTF:

- Modality:
  Structural MRI


- Purpose:
  MRI representation learning


- Subjects:

  326 participants

  - 196 PD
  - 130 Healthy Controls


- Imaging:

  T1-weighted MRI


- Processing:

  - skull stripping
  - bias correction
  - MNI normalization
  - resizing to 224×224 pixels



## 3. Daphnet Freezing-of-Gait Dataset

Role in MRTF:

- Modality:
  Wearable sensor


- Purpose:

  Parkinsonian motor representation learning


- Subjects:

  10 Parkinson's disease patients


Important:

Daphnet does not contain healthy controls.

Therefore, it is NOT used for PD versus HC classification.

The dataset is used only for learning Parkinsonian motor-related latent representations.


Processing:

- 5-second temporal windows
- wavelet denoising
- FFT transformation
- motion feature extraction



# Dataset Organization


After downloading datasets, organize files as:


datasets/

├── raw/

│   ├── pc_gita/

│   ├── ppmi/

│   └── daphnet/


├── processed/

│   ├── voice/

│   ├── mri/

│   └── sensor/


└── metadata/


# Data Partition Protocol


MRTF uses:

Training:
70%

Validation:
15%

Testing:
15%


Partitioning strategy:

Subject-wise splitting


Leakage prevention:

- No participant appears in multiple subsets.
- Normalization parameters are calculated only from training data.
- GAN augmentation is applied only to training data.
- Latent projection parameters are learned only from training embeddings.



# Citation


If MRTF code is used, please cite the corresponding manuscript.

