# TIP Oximeter

## A Deep Learning Framework for SpO2 Estimation from PPG Signals


## Overview

TIP Oximeter is an advanced deep learning framework for estimating blood oxygen saturation (SpO2) from photoplethysmography (PPG) signals. The system utilizes a novel dual-branch neural network architecture (TIPNet) that processes both short-term and long-term temporal features to achieve accurate SpO2 estimation, with optional pressure data integration for improved performance.

## Key Features

- **Dual-Branch Architecture**: Combines short-term and long-term temporal feature extraction for robust SpO2 estimation
- **Pressure Integration**: Optional integration of pressure data to improve estimation accuracy
- **Real-time Capability**: Optimized for efficient inference suitable for embedded systems
- **Comprehensive Evaluation**: Includes tools for performance assessment and visualization

## Repository Structure
TIP_Oximeter/ \
├── data/ # Sample data and dataset organization guidelines \
├── models/ # Pre-trained model weights \
├── src/ # Source code \
│ ├── createTIPNet.m # TIPNet architecture definition \
│ ├── createNet.m # Network branch creation function \
│ ├── getInputSegments.m # Data preprocessing function \
│ ├── processMB.m # Minibatch processing function \
│ ├── modelLoss.m # Loss function definition \
│ └── main.m # Main script for training and evaluation \
├── utils/ # Utility functions \
├── log/ # Training logs and saved models \
├── results/ # Evaluation results and visualizations \
├── docs/ # Documentation \
└── README.md # This file

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/TIP_Oximeter.git
   cd TIP_Oximeter
   ```

2. Ensure you have MATLAB R2022a or later with the following toolboxes:
   - Deep Learning Toolbox
   - Signal Processing Toolbox
   - Statistics and Machine Learning Toolbox

## Usage

### Training a New Model

1. Prepare your dataset following the structure in `data/README.md`
2. Open `src/main.m` and set the following parameters:
   ```matlab
   trainNetworkFlag = 1;        % Enable training mode
   usingPressureFlag = 1;       % Set to 0 if pressure data is unavailable
   datasetFolder = "path/to/your/dataset";
   ```
3. Adjust training hyperparameters as needed
4. Run `main.m` to start training

### Evaluating a Pre-trained Model

1. Open `src/main.m` and set the following parameters:
   ```matlab
   trainNetworkFlag = 0;        % Disable training mode
   usingPressureFlag = 1;       % Must match the model's configuration
   ```
2. Run `main.m` and select a pre-trained model when prompted

## Performance

TIPNet achieves state-of-the-art performance compared to existing methods:

| Method | MAE | RMSE | Bias | SD |
|--------|-----|------|------|-----|
| Static | 0.73 | 0.94 | NA | 0.94 |
| R Model | 0.84 | 1.01 | -0.27 | 0.90 |
| **Proposed** | **0.48*** | **0.75*** | **0.08*** | **0.73*** |
| Proposed (w/o pressure) | 0.52 | 0.82 | 0.13 | 0.78 |

*Results statistically significant (p < 0.05)

All results refer to our forthcoming publication. The proposed TIPNet method demonstrates significant improvements over existing approaches, with the pressure-integrated version showing the best performance across all metrics.

## Citation

If you use TIP Oximeter in your research, please cite our paper (coming soon)