from dataset.text8 import download_text8, load_text8
from dataset.vocabulary import Vocabulary

def get_training_data():
    # download and load the dataset
    path = download_text8()
    tokens = load_text8(path)

    print("Downloaded and loaded text8 dataset, number of tokens:", len(tokens))

    # create the vocabulary 
    print("Crating vocabulary...")
    voc = Vocabulary(tokens)
    print("Vocabulary size:", len(voc.word2idx))


    # encode the tokens with subsampling
    print("Encoding tokens with subsampling...")
    tokens = voc.encode_subsampled(tokens)
    print("Number of tokens after encoding and subsampling:", len(tokens))

    return tokens, voc