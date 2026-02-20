# word2vec-from-scratch

## Goal

Minimal implementation of the Skip-gram Word2Vec model with negative sampling. The project trains word embeddings from scratch on the text8 corpus, handles data prep, training, and basic evaluation utilities with minimal dependencies.

## Dataset

- Uses the text8 corpus (100 MB of Wikipedia text). It is downloaded automatically to data/text8 on first run and optionally truncated via the max_tokens flag for quicker experiments.

## Project layout

```
word2vec-from-scratch/
├─ run.py                   # Typer CLI entry point
├─ environment.yml          # Conda environment spec
├─ data/                    # text8 download cache
├─ output/                  # Saved runs (embeddings, loss curves)
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

1. Install dependencies (recommended via conda using environment.yml) and ensure Python 3.10+.
2. From the repo root, launch the training with:

   ```bash
   python run.py train --run-name test0 --embed-dim 100 --epochs 1 --max-window 5 --n-negatives 5 --max-tokens 1000000
   ```

3. Outputs land in output/<run*name>: embeddings*<run*name>.npz, loss*<run*name>.npz, and loss*<run_name>.png.

## Notebook usage

- Open eval*embeddings.ipynb after a training run. Point it to a saved embeddings*<run_name>.npz to explore nearest neighbours and analogies.

## Design choices

- Skip-gram with negative sampling to keep training efficient on large vocabularies.
- Unigram^{3/4} noise distribution to balance frequent/rare negatives.
- Mikolov subsampling to down-weight extremely frequent tokens and speed convergence.
- Linear learning-rate decay.
- Separate input/output embedding matrices (W_in, W_out) for standard SGNS updates; only W_in used for similarity queries.
- Uniform init for W_in and zero init for W_out.

## Reusing saved embeddings

- Load via MyWord2Vec.load(file) to restore weights and vocabulary, then call get_most_similar or analogy for probing.
