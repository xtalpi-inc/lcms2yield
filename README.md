# lcms2yield
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repository contains the data and code for the paper **"From Purification to Prediction: AI-Powered UV Spectral Prediction as a New Paradigm for Yield Quantification in Modern Chemical Synthesis"**.

## 📦 Installation

### Prerequisites
- Python 3.11
- pip package manager

### Install Dependencies

```bash
# Install PyTorch
pip3 install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# Install main packages
pip install torch_geometric==2.5.3 performer_pytorch==1.1.4 rdkit==2023.9.6

# Install additional dependencies
pip install pykan==0.2.7 pyyaml==6.0.2 matplotlib==3.9.0 pandas==2.3.2 numpy==1.26.4
```

## 📊 Data

This repository includes the following datasets:

- **`data/FDA.csv`**: Contains correction factors and concentration data for FDA-approved molecules
- **`data/Reaction.csv`**: Contains yield data and correction factors for chemical reactions

These datasets are used for training and validation of the AI models in this research.

## 🚀 Quick Start

### Basic Usage

```python
from predict import predict_by_smiles_list

# Example SMILES strings
SMILES_list = [
    'CN(N=C1)C=C1CC2(CC(F)(F)C2)NC(CC3=CC(C=C(Br)C=N4)=C4C=C3)C5=C(O)C=CC=N5', 
    'Nc1[nH]c(=O)nc[c:1]1[N:1]1CSC[C@H]1C(=O)O'
]

# Get predictions
res = predict_by_smiles_list(SMILES_list)
model_pth = res['model_pth']
preds = res['preds']

print('Predicted CFs:', preds)
# Output: Predicted CFs: [1.1300693  0.43467927]
```


## 🔧 API Reference

### `predict_by_smiles_list(smiles_list)`
Predict correction factors for a list of SMILES strings.

**Parameters:**
- `smiles_list` (List[str]): List of SMILES strings

**Returns:**
- `dict`: Dictionary containing:
  - `model_pth` (str): Path to the model used
  - `preds` (List[float]): Predicted correction factors






## 📄 License

This project is licensed under the Apache-2.0 License.



---

