# Connect 4 AI — MCTS-Based Training with CNN & Transformer Models

This repository contains code and saved models for training AI agents to play Connect 4 using two different deep learning architectures — a **Convolutional Neural Network (CNN)** and a **Transformer** model. Training data is generated using Monte Carlo Tree Search (MCTS) self-play at varying search sizes.

---

## Files Included

| File Name                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `connect4_cnn_training.ipynb`     | Trains a CNN-based Connect 4 model using MCTS-generated board states.      |
| `transformer_training.ipynb`      | Trains a Transformer-based model on the same MCTS dataset.                 |
| `connect4_transformer_epoch_25.h5`| Saved Keras model weights after 25 epochs of Transformer training.         |

---

## Project Description

We use MCTS to simulate thousands of Connect 4 games and create labeled training data. This data is used to train two types of deep learning models:

- **CNN Model**: Uses deep residual convolutional layers with squeeze-and-excitation (SE) blocks to capture spatial patterns in the 6x7 board.
- **Transformer Model**: Learns the sequence and board context using attention mechanisms.

Both models aim to predict the best next move for the current board state.

---

##  Libraries Used

- Python
- NumPy
- Pandas
- TensorFlow / Keras
- scikit-learn

---

## How to Use

1. Clone the repo or download the `.ipynb` files.
2. Make sure the dataset file `mcts7500_pool.csv` is in the same directory or modify the path in the notebook.
3. Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn
