# Transformer Implementation from Scratch in PyTorch

This project provides a clean, modular implementation of the original Transformer architecture as described in the paper *"Attention is All You Need"*. It is designed for educational purposes and provides a functional template for sequence-to-sequence tasks.

## 🚀 Key Features

*   **Multi-Head Attention**: Implements scaled dot-product attention with head splitting and combining.
*   **Positional Encoding**: Uses sine and cosine functions to provide sequence order information.
*   **Layer Normalization & Residual Connections**: Standard "Add & Norm" blocks within encoder and decoder layers.
*   **Masking Logic**: Includes padding masks for the encoder and look-ahead (causal) masks for the decoder.

## 🏗️ Architecture Overview

The model consists of the following components:
1.  **MultiHeadAttention**: Parallel attention mechanisms for learning diverse context.
2.  **PositionWiseFeedForward**: A two-layer linear network with ReLU activation applied to each position.
3.  **PositionalEncoding**: Injects relative/absolute position of tokens.
4.  **EncoderLayer/DecoderLayer**: The fundamental building blocks of the Transformer.
5.  **Transformer**: The top-level module coordinating embeddings, encoding, and decoding.
