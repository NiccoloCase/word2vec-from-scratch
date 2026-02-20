from dataset.training import get_training_data
from utils import build_noise_distribution
from model.word2vec import MyWord2Vec
from config import OUTPUT_DIR


def train(run_name: str, embed_dim: int = 100, epochs: int = 1, max_window: int = 5, n_negatives: int = 5):
    print("Training the model...")

    # load training data and build the vocabulary
    tokens, voc = get_training_data()

    # for the sampling of negative examples we build a probability distribution over the vocabulary, based on word frequencies:
    freqs = voc.freqs.values()
    noise_dist = build_noise_distribution(freqs)

    # init model
    model = MyWord2Vec(
        vocab_size = len(voc.word2idx),
        embed_dim = embed_dim,
        noise_dist = noise_dist,
        max_window = max_window,
        n_negatives = n_negatives
    )

    # train the model
    model = model.train(
        train_tokens = tokens,
        noise_dist = noise_dist,
        max_window = max_window,
        epochs = epochs
    )

    # save the resulting embeddings
    output_file = OUTPUT_DIR / f"embeddings_{run_name}.npz"
    model.save_embeddings(voc, output_file)
