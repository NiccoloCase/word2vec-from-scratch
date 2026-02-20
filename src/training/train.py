from dataset.training import get_training_data

def train():
    print("Training the model...")
    tokens = get_training_data()
    print(f"Number of tokens: {len(tokens)}")
    # Here you would add the actual training code for your Word2Vec model

