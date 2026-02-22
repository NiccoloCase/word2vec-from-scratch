# Word2Vec Training From Scratch

## Goal

Minimal implementation of the Skip-gram Word2Vec model with negative sampling. The project trains word embeddings from scratch on the text8 corpus, handles data prep, training, and basic evaluation utilities with minimal dependencies (no PyTorch/TensorFlow, only NumPy).

## Dataset

- Uses the text8 corpus (Wpedia text). It is downloaded automatically to data/text8 on first run and optionally truncated via the max_tokens flag for quicker experiments (I used only the first 5M tokens for testing).

## Project layout

```
word2vec-from-scratch/
├─ run.py                   # Typer CLI entry point
├─ environment.yml          # Conda environment spec
├─ data/                    # text8 download cache
├─ output/                  # Saved runs (embeddings, loss curves)
├─ jobs/                    # Slurm job scripts for training
├─ src/
│  ├─ config.py             # Paths and directory bootstrap
│  ├─ utils.py              # Tokeniser, noise dist, sigmoid, LR decay
│  ├─ dataset/
│  │  ├─ text8.py           # Download/load text8
│  │  ├─ vocabulary.py      # Vocab build + subsampling
│  │  └─ training.py        # Data prep pipeline
│  ├─ model/
│  │  └─ word2vec.py        # SGNS model, training, eval utils
│  └─ training/
│     └─ train.py           # Experiment runner and logging
└─ eval_embeddings.ipynb    # Inspect embeddings/analogies/loss
```

## How to train

1. Install dependencies using environment.yml
2. From the repo root, launch the training with:

   ```bash
   python run.py train --run-name test0 --embed-dim 100 --epochs 1 --max-window 5 --n-negatives 5 --max-tokens 1000000
   ```

3. Outputs land in output/<run*name>: embeddings*<run*name>.npz, loss*<run*name>.npz, and loss*<run_name>.png.

## Notebook usage

- Open eval*embeddings.ipynb after a training run. Point it to a saved embeddings*<run_name>.npz to explore nearest neighbours and analogies.

## Design choices

- Skip-gram with negative sampling to keep training efficient on large vocabularies
- Unigram^{3/4} noise distribution to balance frequent/rare negatives
- Mikolov subsampling to down-weight extremely frequent tokens
- Linear learning-rate decay.
- Separate input/output embedding matrices (W_in, W_out) for standard SGNS updates; but only W_in used for similarity queries
- Uniform init for W_in and zero init for W_out
