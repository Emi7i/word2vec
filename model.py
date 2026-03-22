import os
import json

import numpy as np

from tokenizer import Tokenizer


class Model:

    def __init__(self,
                 tk: Tokenizer,
                 embedding_dim: int,
                 window_size: int = 5):
        self.tokenizer = tk
        self.window_size = window_size
        self.vocabulary_size = len(self.tokenizer.index2word)
        self.embedding_dim = embedding_dim
        self.current_epoch = 0

        # Input weight matrix
        self.C = np.random.rand(self.vocabulary_size, embedding_dim)
        # Output weight matrix
        self.W = np.random.rand(self.vocabulary_size, embedding_dim)

    def train(self, epochs: int, learning_rate: float):
        """Train the model by iterating over every word for a given number of epochs,
        performing a forward pass, computing the loss, and updating the weights via backpropagation.
        """
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            total_loss = 0.0
            for i, word in enumerate(self.tokenizer.train):
                target_index = self.tokenizer.word2index[word]
                probs, surround_words, context_vec = self.forward_pass(i)
                loss = self.loss(target_index, probs)
                total_loss += loss
                self.backward_pass(target_index, probs, surround_words, context_vec, learning_rate)
                print(f"\rEpoch {epoch} | Word {i}/{len(self.tokenizer.train)}"
                      f" | Loss: {total_loss / (i + 1):.4f}", end="")
            self.save(loss = total_loss / len(self.tokenizer.train), learning_rate=learning_rate)
            print()

    def validate(self):
        """Validate the model by predicting the top 5 most likely words."""
        # range(len(valid_tk.words))
        for i in range(0, 100, 5):
            self.predict(i, "Valid")

    def predict(self, target_index: int, mode: str = "Train"):
        """Predict the top 5 most likely words."""
        # Let's assume the tokenizer has all the words from the validation file

        probs, surround_words, context_vec = self.forward_pass(target_index, mode)
        top_indices = np.argsort(probs)[::-1][:5]
        print(f"{[self.tokenizer.index2word[i] for i in surround_words]}: ")
        print([self.tokenizer.index2word[i] for i in top_indices])

    def forward_pass(self,
                     target_index: int,
                     mode: str = "Train") -> tuple[np.ndarray, list[int], np.ndarray]:
        """Compute the scores and probabilities for the surrounding words given the center word.
        Get the average of the context word vectors,
        Multiply by the output weight matrix.
        """
        surround_words = self.get_context_words(target_index, mode)
        context_vec = self.average_words(surround_words)
        scores = context_vec @ self.W.T
        probs = self.softmax(scores)
        return probs, surround_words, context_vec

    @staticmethod
    def loss(target_index: int, probs: np.ndarray) -> float:
        """Compute cross-entropy loss: -ln(e^(W[target] · h) / ∑_w e^(W[w] · h))."""
        loss = -np.log(probs[target_index])
        return loss

    def backward_pass(self,
                      target_index: int,
                      probs: np.ndarray,
                      surround_words: list[int],
                      vector: np.ndarray,
                      learning_rate: float):
        """Compute gradients and update weights."""
        dL_dy = probs.copy()
        dL_dy[target_index] -= 1
        dL_dx = self.W.T @ dL_dy # we are using probs which are already averaged

        self.W -= learning_rate * np.outer(dL_dy, vector)
        for index in surround_words:
            self.C[index] -= learning_rate * dL_dx  # update context rows

    def average_words(self, indexes: list[int]) -> np.ndarray:
        """Compute the average of the context word vectors."""
        return self.C[indexes].mean(axis=0)

    def get_context_words(self, target_index: int, mode: str = "Train") -> list[int]:
        """Get the surrounding words around the target word and returns their indexes."""
        if mode == "Train":
            wordlist = self.tokenizer.train
        elif mode == "Valid":
            wordlist = self.tokenizer.valid
        elif mode == "Test":
            wordlist = self.tokenizer.test
        else:
            raise ValueError("Invalid mode")

        start = max(0, target_index - self.window_size)
        end = min(len(wordlist), target_index + self.window_size + 1)

        context = wordlist[start:target_index] + wordlist[target_index+1:end] # removes only the center word (only once)
        return [self.tokenizer.word2index[w] for w in context] # returns indexes of words

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def save(self, path: str = "model", loss: float = 0.0, learning_rate: float = 0.0):
        """Save the model weights and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/C.npy", self.C)
        np.save(f"{path}/W.npy", self.W)
        with open(f"{path}/metadata.json", "w") as f:
            json.dump({
                "epoch": self.current_epoch,
                "embedding_dim": self.embedding_dim,
                "window_size": self.window_size,
                "loss": f"{loss:.4f}",
                "learning_rate": f"{learning_rate}",
            }, f)
        print("\nModel saved!")

    def load(self, path: str = "model"):
        """Load the model from the specified path."""
        self.C = np.load(f"{path}/C.npy")
        self.W = np.load(f"{path}/W.npy")

        with open(f"{path}/metadata.json", "r") as f:
            metadata = json.load(f)

        self.current_epoch = metadata["epoch"]
        # If embedded dimensions do not match, override them and notify user
        embedded_dims = metadata["embedding_dim"]
        if embedded_dims != self.embedding_dim:
            print(f"Warning: embedding dimension does not match. Overriding embedded dimensions!"
                  f"\nNew dimensions: {embedded_dims}\n")
            self.embedding_dim = embedded_dims

        print(f"Model loaded, resuming from epoch {self.current_epoch}")