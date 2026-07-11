# RoBERTa

This repository contains an improved multi-label text classification model based on RoBERTa.

It features an advanced architecture including Token-level Attention Pooling, Label-Embeddings, Asymmetric Loss (ASL), and an optional Multi-Label Supervised Contrastive Loss.

## Project Structure

The code is organized modularly for readability and maintainability:

```text
RoBERTa/
├── config.py           # Hyperparameters, paths, and label definitions
├── dataset.py          # GoEmotions data loading and tokenization
├── losses.py           # Asymmetric Loss and Contrastive Loss implementations
├── metrics.py          # Multi-label evaluation metrics (F1, Precision, Recall)
├── model.py            # Improved RoBERTa architecture with custom heads
├── trainer.py          # Training loop, evaluation, and threshold optimization
└── main.py             # Main entry point to execute the pipeline
