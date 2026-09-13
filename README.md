<p align="center">
  <img src="https://github.com/user-attachments/assets/436635dc-0d6f-411a-9e55-73eb2bbc5a46" width="512" heigth="120">
</p>


<p align="center">
  A data compression framework developed as a project for the Compressione Dati (Data Compression) course, part of the Computer Science Master's Degree program at the University of Salerno
</p>


## Table of Contents
- [Overview](#Overview)
- [Features](#Features)
- [Results](#Results)
- [Requirements](#Requirements)
- [How to replicate](#How-to-replicate)


## Overview 
  This project builds on an existing architecture for semantic coding described by Hao et al. (https://arxiv.org/abs/2505.08536), implementing that architecture and extending it 
  with the addition of
  BCH codes with shorter parameters. Short BCH codes are used to encode and decode segments of phrases in parallel, cutting down on latency by processing multiple segments 
  simultaneously rather than sequentially. Any residual errors left uncorrected by the BCH codes are then resolved by an AI model, which leverages the semantic context of the 
  content to identify and fix these errors as a semantic task, rather than relying on additional redundancy.


## Features
1) Shorter BCH parameters (32,16)
2) Semantic correction module with BART model + SuperBPE tokenizer
3) Alternative semantic correction module with T5 model


## Results
| Approach | BLER | BLEU | ROUGE-L |
| ------------- | ------------- | ------------- | ------------- |
| Original approach replication | 0.48 | 0.55 | 0.75 |
| BART + SuperBPE | 1.00 | 0.15 | 0.35 |
| T5 | 0.46 | 0.64 | 0.84 |


## Requirements 
- Python dependencies are listed in the "requirements.txt" file


## How to replicate
1) Clone the repository
```bash
git clone https://github.com/raffaeleav/detectwins.git
```
2) Install dependencies (assuming conda is being used)
```bash
conda create -n "ssbsc" python=3.10 
conda activate ssbsc
pip install -r ssbsc/requirements.txt
```
3) Install Cuda toolkit (assuming you have Ubuntu 22.04)
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.runsudo
sh cuda_12.1.0_530.30.02_linux.run
pip install torch torchvision torchaudio
```
4) Run models training and testing
```bash
cd ssbsc
python main.py
```
