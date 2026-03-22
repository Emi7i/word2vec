from tokenizer import Tokenizer
from model import Model


# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
WINDOW_SIZE = 5
NEGATIVE_SAMPLES_NUM = 7

# Dataset
DATASET_PATH = "textFile.txt"
FREQUENCY = 2


if __name__ == "__main__":
    tk = Tokenizer(DATASET_PATH, FREQUENCY)
    print("Processed file~!")
    model = Model(tk, EMBEDDING_DIM, WINDOW_SIZE)
    model.load()
    print(model.most_similar("holmes"))
    print(model.most_similar("watson"))
    print(model.most_similar("london"))
    # print("Training model!")
    # model.train(EPOCHS, LEARNING_RATE, NEGATIVE_SAMPLES_NUM)
    # print("Validating!")
    # model.validate()
