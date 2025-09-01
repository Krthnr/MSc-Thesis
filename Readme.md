# Membership Inference Attacks on Diffusion Models (DP-Promise & DCTDiff)

This repository contains the implementation, training, and evaluation pipeline for analyzing **privacy leakage in diffusion models**. Two representative models are studied:

- **DP-Promise** – a diffusion model trained with differential privacy guarantees.  
- **DCTDiff** – a non-private baseline diffusion model.  

The repository also includes **black-box and white-box membership inference attack (MIA) evaluations**, consistent with the experiments reported in the thesis.

---

## Repository Structure

- `DP-Promise_Model_Eval.ipynb` – Training and evaluation pipeline for DP-Promise.  
- `DCTDiff_Model_Eval.ipynb` – Training and evaluation pipeline for DCTDiff.  
- `DP-Promise_Black_Box_Attack.ipynb` – Black-box MIAs on DP-Promise.  
- `DCTDiff_Black_Box_Attack.ipynb` – Black-box MIAs on DCTDiff.  
- `DP-Promise_White_Box_Attack.ipynb` – White-box MIAs on DP-Promise.  

All **trained checkpoints, processed datasets, and generated samples (.npz)** are hosted on Google Drive. Configurations and evaluation results are included in this repository for reproducibility.

---

## Model Setup

### DP-Promise
1. Clone the original [DP-Promise GitHub repository](#).  
2. Install required packages from `requirements.txt`.  
3. **Data processing**:  
   - Download CelebA dataset (open-source from Kaggle).  
   - Preprocess into $32 \times 32$ and $64 \times 64$ resolutions.  
   - Apply 80/20 train–test split (train/test sets hosted on Google Drive).  
4. **Vanilla Pretraining**:  
   - Train vanilla diffusion model on CelebA (32, 64).  
   - Store checkpoints for use in DP-Promise configuration files.  
5. **DP Training**:  
   - Fine-tune with DP-Promise using pretrained vanilla checkpoints.  
   - Train three models per resolution with privacy budgets $\epsilon \in \{1,5,10\}$.  
   - Generate synthetic images for evaluation.  
6. **Evaluation**:  
   - Install `requirements_eval.txt`.  
   - Compute FID and IS between generated images and training data.  
   - Results and config files are provided under `Evaluation/` and `config/`.  

---

### DCTDiff
1. Use the preprocessed CelebA $64 \times 64$ dataset from DP-Promise.  
2. Install packages from notebook cell (`DCTDiff_Model_Eval.ipynb`).  
3. Train model using provided job setup.  
4. Evaluate generated images using FID across three samplers:  
   - DPM-Solver, Euler ODE, Euler SDE.  
5. Generate samples with all samplers at varying **Number of Function Evaluations (NFE)**: 10, 20, 50, 100.  
6. Adopt $NFE=100$ for attack experiments.  

All checkpoints, generated samples (.npz), and evaluations are provided in Google Drive.

---

## Attack Frameworks

### Black-Box Attacks
- **DP-Promise (`DP-Promise_Black_Box_Attack.ipynb`)**:  
  - Baseline: Raw pixel similarity (cosine, Euclidean).  
  - Feature-based: CLIP embeddings + LPIPS perceptual distance.  
  - Experiments:  
    - CelebA 32 & 64 with seeds 42 and 1234.  
    - Scale: 1k, 10k, and 60k generated images vs. member/non-member splits.  
    - Median results across 10 random seeds with ROC–AUC curves and histograms.  

- **DCTDiff (`DCTDiff_Black_Box_Attack.ipynb`)**:  
  - Convert `.npz` to `.png` for attacks.  
  - CLIP + LPIPS features across samplers (DPM, ODE, SDE).  
  - Seeds 42 and 1234 + median results over 10 seeds with ROC–AUC and score histograms.  

---

### White-Box Attacks
- **DP-Promise (`DP-Promise_White_Box_Attack.ipynb`)**:  
  - Loss-based scoring using per-sample denoising predictions.  
  - Experiments for CelebA 32 and 64 across $\epsilon \in \{1,5,10\}$.  
  - Outputs:  
    - ROC–AUC across timesteps.  
    - Histograms of member vs. non-member loss distributions.  
    - Reconstruction trajectories (member vs. non-member).  

---

## Data & Checkpoints

- **Processed CelebA (32, 64)** – Train/test splits (80/20).  
- **Vanilla checkpoints** – Pretraining for DP-Promise.  
- **DP-Promise checkpoints** – Trained models for $\epsilon \in \{1,5,10\}$.  
- **DCTDiff checkpoints** – Trained for 400K steps.  
- **Generated samples** – Stored as `.npz` files.  
- **Evaluation outputs** – IS/FID results and attack plots.  

All are hosted in Google Drive (links available in the repository).

---

## Reproducibility Notes
- Config files for all training runs are under `config/`.  
- Evaluation results (FID, IS, ROC–AUC, histograms) are under `Evaluation/`.  
- Attacks follow the sequence in respective notebooks and reproduce results from the thesis.  

---

## Citation
If you use this repository or the methodology in your work, please cite:  

