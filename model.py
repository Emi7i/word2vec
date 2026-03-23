import os
import json
import random

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

    def train(self, epochs: int, learning_rate: float, negative_samples_num: int):
        """Train the model by iterating over every word for a given number of epochs,
        performing a forward pass, computing the loss, and updating the weights via backpropagation.
        """
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            total_loss = 0.0
            for i, word in enumerate(self.tokenizer.words):
                target_index = self.tokenizer.word2index[word]
                scores, surround_words, context_vec = self.forward_pass(i)
                loss, negative_indices = self.loss(target_index, scores, negative_samples_num)
                total_loss += loss
                self.backward_pass(target_index, surround_words, context_vec, learning_rate, negative_indices)
                print(f"\rEpoch {epoch} | Word {i}/{len(self.tokenizer.words)}"
                      f" | Loss: {total_loss / (i + 1):.4f}", end="")
            self.save(loss = total_loss / len(self.tokenizer.words), learning_rate=learning_rate)
            print()

    def forward_pass(self, target_index: int) -> tuple[np.ndarray, list[int], np.ndarray]:
        """Compute the scores and probabilities for the surrounding words given the center word.
        Get the average of the context word vectors,
        Multiply by the output weight matrix.
        """
        surround_words = self.get_context_words(target_index)
        context_vec = self.average_words(surround_words)
        scores = context_vec @ self.W.T
        return scores, surround_words, context_vec


    def loss(self, target_index: int, scores: np.ndarray, negative_samples_num: int) -> tuple[float, list[int]]:
        """Negative sampling loss: L = log σ(v'_target · h) + Σ log σ(-v'_negative · h)."""
        # positive sample
        loss = -np.log(self.sigmoid(scores[target_index]))

        # negative samples (do not include the target word)
        negative_indices = random.sample(
            [i for i in range(self.vocabulary_size) if i != target_index],
            negative_samples_num
        )

        for idx in negative_indices:
            loss -= np.log(1 - self.sigmoid(scores[idx]))

        return float(loss), negative_indices

    def backward_pass(self,
                      target_index: int,
                      surround_words: list[int],
                      vector: np.ndarray,
                      learning_rate: float,
                      negative_indices):
        """Compute gradients and update weights.

        Based on Bengio et al. (2003), Section 2, Backward/Update Phase:
        - dL_dy_pos: ∂L/∂y_j = 1(j==wt) - p_j for target word
        - dL_dy_neg: ∂L/∂y_j = 0 - p_j for negative samples
        - W updates: W_j ← W_j + ε * ∂L/∂y_j * x
        - dL_dx: ∂L/∂x accumulated from target and negative words
        - C updates: C(wt-k) ← C(wt-k) + ε * ∂L/∂x(k)
        """
        dL_dy_pos = self.sigmoid(self.W[target_index] @ vector) - 1

        dL_dy_neg = self.sigmoid(self.W[negative_indices] @ vector)

        # Update W for target and negative words only
        self.W[target_index] -= learning_rate * dL_dy_pos * vector
        self.W[negative_indices] -= learning_rate * dL_dy_neg[:, np.newaxis] * vector

        dL_dx = dL_dy_pos * self.W[target_index] + np.sum(dL_dy_neg[:, np.newaxis] * self.W[negative_indices], axis=0)

        for index in surround_words:
            self.C[index] -= learning_rate * dL_dx

    def average_words(self, indexes: list[int]) -> np.ndarray:
        """Compute the average of the context word vectors."""
        return self.C[indexes].mean(axis=0)

    def get_context_words(self, target_index: int) -> list[int]:
        """Get the surrounding words around the target word and returns their indexes."""
        start = max(0, target_index - self.window_size)
        end = min(len(wordlist), target_index + self.window_size + 1)

        context = wordlist[start:target_index] + wordlist[target_index+1:end] # removes only the center word (only once)
        return [self.tokenizer.word2index[w] for w in context] # returns indexes of words

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute sigmoid values for each sets of scores in x."""
        return np.clip(1 / (1 + np.exp(-x)), 1e-7, 1 - 1e-7) # clip so log doesn't blow up

    def most_similar(self, word: str, top_n: int = 5) -> list[str]:
        """Find most similar words using cosine similarity."""
        if word not in self.tokenizer.word2index:
            print(f"Word '{word}' not in vocabulary")
            return []

        word_index = self.tokenizer.word2index[word]
        word_vector = self.C[word_index]

        similarities = self.C @ word_vector / (
                np.linalg.norm(self.C, axis=1) * np.linalg.norm(word_vector)
        )

        top_indices = np.argsort(similarities)[::-1][1:top_n + 1]
        return [self.tokenizer.index2word[i] for i in top_indices]

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