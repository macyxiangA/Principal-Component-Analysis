# PCA and N-Gram Language Model

This project contains two components implemented in Python:

1. A Principal Component Analysis (PCA) module for dimensionality reduction and image reconstruction.
2. A character-level N-gram language model supporting probability computation and text generation.

## PCA Functions

- load_and_center_dataset(filename): Loads data and centers it by subtracting the mean.
- get_covariance(dataset): Computes the sample covariance matrix.
- get_eig(S, k): Returns the top k eigenvalues and eigenvectors.
- get_eig_prop(S, prop): Returns eigenvectors explaining more than a given variance proportion.
- project_and_reconstruct_image(image, U): Projects an image into PCA subspace and reconstructs it.
- display_image(...): Displays original and reconstructed images side by side.

## N-Gram Language Model

- fit(text): Builds n-gram counts from training text.
- logprob(s): Computes log-probability of a string.
- prob(s): Computes string probability.
- next_char_distribution(context): Returns the next-character distribution.
- generate(num_chars, seed): Generates text from the model.

## How to Use

Place your `.npy` dataset or text input in the project directory and call the appropriate functions.

## File Structure

- pca_and_ngram.py — main implementation file
- dataset.npy — sample data (optional)
- README.md

## Requirements

Python 3  
NumPy  
SciPy  
Matplotlib

## Author

Macy Xiang  
https://github.com/macyxiangA
