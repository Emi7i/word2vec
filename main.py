from tokenizer import Tokenizer
from model import Model

EPOCHS = 50
LEARNING_RATE = 0.01
EMBEDDING_DIM = 300
WINDOW_SIZE = 5

if __name__ == "__main__":
    tk = Tokenizer("textFile.txt")
    model = Model(tk, EMBEDDING_DIM, WINDOW_SIZE)
    model.train(EPOCHS, LEARNING_RATE)
