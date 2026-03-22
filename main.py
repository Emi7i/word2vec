from tokenizer import Tokenizer
from model import Model


# Hyperparameters
EPOCHS = 4
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
WINDOW_SIZE = 5


if __name__ == "__main__":
    tk = Tokenizer("textFile.txt")
    print("Processed file~!")
    model = Model(tk, EMBEDDING_DIM, WINDOW_SIZE)
    model.load()
    print("Training model!")
    model.train(EPOCHS, LEARNING_RATE)
    # print("Validating!")
    # model.validate()
