import numpy as np
from tokenizer import Tokenizer

class Model:

    def __init__(self, tk: Tokenizer, embedding_dim: int, window_size: int = 5):
        self.tokenizer = tk
        self.window_size = window_size
        self.vocabulary_size = len(self.tokenizer.index2word)
        self.embedding_dim = embedding_dim

        # Matrices
        self.C = np.random.rand(self.vocabulary_size, embedding_dim)  # This is our input weight matrix
        self.W = np.random.rand(self.vocabulary_size, embedding_dim)  # This is our output weight matrix

    def train(self, epochs: int, learning_rate: float):
        """
        Trains the model by iterating over every word for a given number of epochs,
        performing a forward pass, computing the loss, and updating the weights via backpropagation.
        """
        for epoch in range(epochs):
            total_loss = 0
            for i, word in enumerate(self.tokenizer.words):
                target_index = self.tokenizer.word2index[word]
                probs, surround_words, context_vec = self.forward_pass(i)
                loss = self.loss(target_index, probs)
                total_loss += loss
                self.backward_pass(target_index, probs, surround_words, context_vec, learning_rate)
            print(f"Epoch {epoch} Loss: {total_loss / len(self.tokenizer.words)}")

    def forward_pass(self, target_index: int) -> tuple[np.ndarray, list[int], np.ndarray]:
        """
        Computes the scores and probabilities for the surrounding words given the center word
        Gets the average of the context word vectors,
        Multiplies by the output weight matrix
        """
        surround_words = self.get_context_words(target_index)
        context_vec = self.average_words(surround_words)
        scores = context_vec @ self.W.T
        probs = self.softmax(scores)
        return probs, surround_words, context_vec

    @staticmethod
    def loss(target_index: int, probs: np.ndarray) -> float:
        """ Compute cross-entropy loss: -ln(e^(W[target] · h) / ∑_w e^(W[w] · h) ) """
        loss = -np.log(probs[target_index])
        return loss

    def backward_pass(self, target_index: int, probs: np.ndarray, surround_words: list[int], vector: np.ndarray, learning_rate: float):
        """ Compute gradients and update weights """
        dL_dy = probs.copy()
        dL_dy[target_index] -= 1
        dL_dx = (self.W.T @ dL_dy) # we are using probs which are already averaged

        self.W -= learning_rate * np.outer(dL_dy, vector)
        for index in surround_words:
            self.C[index] -= learning_rate * dL_dx  # update context rows

    def average_words(self, indexes: list[int]) -> np.ndarray:
        """ Compute the average of the context word vectors """
        num_of_words = len(indexes)
        result = np.zeros(self.embedding_dim)
        for i in indexes:
            result += self.C[i]

        return result / num_of_words

    def get_context_words(self, target_index: int) -> list[int]:
        """ Gets the surrounding words around the target word and returns their indexes """
        start = max(0, target_index - self.window_size)
        end = min(len(self.tokenizer.words), target_index + self.window_size + 1)

        context = self.tokenizer.words[start:target_index] + self.tokenizer.words[target_index+1:end] # removes only the center word (only once)
        return [self.tokenizer.word2index[w] for w in context]   # returns indexes of words

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """ Compute softmax values for each sets of scores in x """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()