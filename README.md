# Connect 4 AI — MCTS-Based Training with CNN & Transformer Models

This repository contains code, data, and saved models for training AI agents to play Connect 4 using two deep learning architectures — a Convolutional Neural Network (CNN) and a Transformer model. Training data is generated using Monte Carlo Tree Search (MCTS) self-play at varying search depths.

---

## Files Included

| File Name                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `connect4_cnn_training.ipynb`     | Trains a CNN-based Connect 4 model using MCTS-generated board states.      |
| `transformer_training.ipynb`      | Trains a Transformer-based model on the same MCTS dataset.                 |
| `connect4_transformer_epoch_25.h5`| Saved Keras model weights after 25 epochs of Transformer training.         |
| `mcts7500_pool.csv`               | Training dataset generated from 7,500 MCTS simulations.                    |

---

## Project Description

We use MCTS to simulate thousands of Connect 4 games and generate labeled training data. This data is used to train two deep learning models:

- CNN Model: Utilizes deep residual convolutional layers with squeeze-and-excitation (SE) blocks to capture spatial patterns in the 6x7 board.
- Transformer Model: Leverages attention mechanisms to model sequential and spatial relationships in the board state.

The models are trained to predict the most optimal column to play next.

To make the trained model accessible outside of a notebook environment, it was deployed as a lightweight API using Flask. The API was containerized using Docker and deployed on an AWS Lightsail instance. This allowed for real-time move predictions by sending board states to the hosted model from external applications.

---

## Libraries Used

- Python
- NumPy
- Pandas
- TensorFlow / Keras
- scikit-learn

---

## How to Use

1. Clone the repo or download the `.ipynb` files.
2. Ensure the dataset file `mcts7500_pool.csv` is present in the directory.
3. Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn
