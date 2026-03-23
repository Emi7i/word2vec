from tokenizer import Tokenizer
from model import Model


# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
WINDOW_SIZE = 5
NEGATIVE_SAMPLES_NUM = 7

# Dataset
DATASET_PATH = "text8.txt"
FREQUENCY = 5
MAX_VOCABULARY_SIZE = 1000000

if __name__ == "__main__":
    tk = Tokenizer(DATASET_PATH, FREQUENCY, MAX_VOCABULARY_SIZE)
    print("Processed file~!")
    model = Model(tk, EMBEDDING_DIM, WINDOW_SIZE)
    model.load()
    print("Training model!")
    model.train(EPOCHS, LEARNING_RATE, NEGATIVE_SAMPLES_NUM)