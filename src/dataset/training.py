from dataset.text8 import download_text8, load_text8


def get_training_data():
    path = download_text8()
    tokens = load_text8(path)
    return tokens