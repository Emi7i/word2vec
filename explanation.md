# Understanding word2vec

## Tokenizer

First we need to load the words from the file. To do this, we created the `Tokenizer` class.

```python
from tokenizer import Tokenizer

# Dataset
DATASET_PATH = "text8.txt"
FREQUENCY = 5
MAX_VOCABULARY_SIZE = 750000

tk = Tokenizer(DATASET_PATH, FREQUENCY, MAX_VOCABULARY_SIZE)
```

```
Vocab size: 11442
```

**Tokenizer** class loads the words from the file and deletes the characters in the *PATTERN* filter. Then it uses the `tokenize` function to map all words into `word2index` and `index2word` dictionaries.

We want to map the words onto numbers so we can use them as indexes for our matrices.

```python
class Tokenizer:
    PATTERN = re.compile(r'[!"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~\t\n]')

    def __init__(self, text_file: str, frequency: int = 0, max_vocabulary_size: int = 0):
        self.text_file = text_file
        self.words = []

        self.word2index = {}
        self.index2word = {}

        self.load_words(frequency, max_vocabulary_size)
        self.tokenize(self.words)
```

---

## Model

Our model is defined in the `Model` class.

```python
from model import Model

# Hyperparameters
EPOCHS = 2
LEARNING_RATE = 0.005
EMBEDDING_DIM = 50
WINDOW_SIZE = 2
NEGATIVE_SAMPLES_NUM = 5

model = Model(tk, EMBEDDING_DIM, WINDOW_SIZE)
```

Let's see what the model actually does when training.

```python
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
```

In each epoch the model calls:
- Forward pass
- Loss
- Backward pass

---

## Forward Pass

*Based on Bengio et al. (2003), Section 2, Forward Phase.*

```python
def forward_pass(self, target_index: int) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Compute the scores and probabilities for the surrounding words given the center word.
    Get the average of the context word vectors,
    Multiply by the output weight matrix.
    """
    surround_words = self.get_context_words(target_index)
    context_vec = self.average_words(surround_words)
    scores = context_vec @ self.W.T
    return scores, surround_words, context_vec
```

**Forward pass** gets the context words around our target word. Then it calculates the context vector by averaging the values in our **input matrix C** for each of the words in `surround_words`. Then it calculates the score by multiplying this vector with transposed output matrix **W**.

### Steps as done in the paper

#### Step (a) — Word feature lookup

$$x(k) \leftarrow C(w_{t-k})$$

Look up the embedding vector for each context word from the input matrix $C$. Each context word is represented as a row in $C$, retrieved by its vocabulary index.

```python
surround_words = model.get_context_words(i)
context_vec = model.average_words(surround_words)  # x = average of x(k) for all k
```

#### Step (b) — Hidden layer

CBOW removes the hidden layer entirely — this is the key simplification from Mikolov et al. (2013) over Bengio et al. (2003). Removing the non-linear hidden layer drastically reduces computational cost while retaining the ability to learn meaningful word embeddings.

#### Step (c/d/e) — Output scores and normalization

Compute a score for every word in the vocabulary, then normalize into probabilities:

$$y_j = x \cdot W_j$$

$$p_j = \frac{e^{y_j}}{S}$$

```python
scores = context_vec @ model.W.T  # y_j for all words in vocabulary
probs = model.softmax(scores)     # p_j = e^y_j / S
```

> **Note:** During training we replace softmax with negative sampling for efficiency. Softmax is shown here to illustrate the mathematical formulation from the paper.

---

## Loss

The loss function measures how wrong the model's prediction is. The lower the loss, the better the model.

```python
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
```

Instead of computing softmax over the entire vocabulary, we use **negative sampling** which turns the problem into binary classification.

Sigmoid outputs a value between 0 and 1, which we interpret as a probability:
- **Target word:** we want $\sigma(\text{score}) \to 1$ (high probability = correct word)
- **Negative words:** we want $\sigma(\text{score}) \to 0$ (low probability = incorrect word)

We use sigmoid instead of softmax because it computes probability for each word independently, allowing us to implement negative sampling and significantly reduce computational cost from $O(V)$ to $O(k)$.

The paper maximizes log-likelihood, but the code **minimizes** the negated loss:

$$\mathcal{L}_{\text{paper}} = \log \sigma(v'_{\text{target}} \cdot h) + \sum_{j} \log \sigma(-v'_{\text{negative}_j} \cdot h)$$

$$\mathcal{L}_{\text{code}} = -\log \sigma(v'_{\text{target}} \cdot h) - \sum_{j} \log \sigma(-v'_{\text{negative}_j} \cdot h)$$

```python
loss = -np.log(model.sigmoid(scores[target_index]))
loss -= np.log(1 - model.sigmoid(scores[negative_index]))
```

---

## Backward Pass

*Based on Bengio et al. (2003), Section 2, Backward/Update phase.*

```python
def backward_pass(self,
                  target_index: int,
                  surround_words: list[int],
                  vector: np.ndarray,
                  learning_rate: float,
                  negative_indices):
    """Compute gradients and update weights."""
    dL_dy_pos = self.sigmoid(self.W[target_index] @ vector) - 1
    dL_dy_neg = self.sigmoid(self.W[negative_indices] @ vector)

    # Update W for target and negative words only
    self.W[target_index] -= learning_rate * dL_dy_pos * vector
    self.W[negative_indices] -= learning_rate * dL_dy_neg[:, np.newaxis] * vector

    dL_dx = dL_dy_pos * self.W[target_index] + np.sum(dL_dy_neg[:, np.newaxis] * self.W[negative_indices], axis=0)

    for index in surround_words:
        self.C[index] -= learning_rate * dL_dx
```

### Steps as done in the paper

#### Step (a) — Perform backward gradient computation

> **Note:** We are using negative sampling instead of full softmax — only the target word and $k$ negative samples are updated per step.

**i) Compute gradients for target and negative words:**

$$\frac{\partial \mathcal{L}}{\partial y_{\text{pos}}} = \sigma(W_t \cdot x) - 1$$

$$\frac{\partial \mathcal{L}}{\partial y_{\text{neg}}} = \sigma(W_j \cdot x)$$

```python
dL_dy_pos = self.sigmoid(self.W[target_index] @ vector) - 1
dL_dy_neg = self.sigmoid(self.W[negative_indices] @ vector)
```

**ii) Update output matrix W for target and negative words only:**

$$W_t \leftarrow W_t - \varepsilon \cdot \frac{\partial \mathcal{L}}{\partial y_{\text{pos}}} \cdot x$$

$$W_j \leftarrow W_j - \varepsilon \cdot \frac{\partial \mathcal{L}}{\partial y_{\text{neg}}} \cdot x$$

```python
self.W[target_index] -= learning_rate * dL_dy_pos * vector
self.W[negative_indices] -= learning_rate * dL_dy_neg[:, None] * vector
```

#### Step (b) — Sum and share $\partial\mathcal{L}/\partial x$

Skipped — this is already handled in step (a).

#### Step (c) — Backpropagate gradient to input vector

Accumulate gradient contributions from both the target word and all negative samples (matching the paper's accumulation form):

$$\frac{\partial \mathcal{L}}{\partial x} \leftarrow \frac{\partial \mathcal{L}}{\partial x} + \frac{\partial \mathcal{L}}{\partial y_j} \cdot W_j \quad \text{(for each } j \text{ in target + negative samples)}$$

Which resolves to:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y_{\text{pos}}} \cdot W_t + \sum_{j} \frac{\partial \mathcal{L}}{\partial y_j} \cdot W_j$$

```python
dL_dx = dL_dy_pos * self.W[target_index] + np.sum(dL_dy_neg[:, None] * self.W[negative_indices], axis=0)
```

#### Step (d) — Update input embeddings C

As written in the paper (gradient **ascent**, maximizing log-likelihood):

$$C(w_{t-k}) \leftarrow C(w_{t-k}) + \varepsilon \cdot \frac{\partial \mathcal{L}}{\partial x^{(k)}}$$

> **Note on sign:** The paper maximizes log-likelihood with `+ε`, while the code minimizes a loss with `−ε`. Both are equivalent — the gradients `dL_dy_pos` and `dL_dy_neg` in the code already reflect a negated loss, so the `−ε` in the code corresponds to `+ε` in the paper's notation.

```python
for index in surround_words:
    self.C[index] -= learning_rate * dL_dx
```

---

# Training the Model and Results

```python
model.train(EPOCHS, LEARNING_RATE, NEGATIVE_SAMPLES_NUM)
```

When training the model on the first 750k words from the text8 dataset, we get the following output:

```
Vocab size: 13966
Processed file~!
Training model!
Epoch 0 | Word 750488/750489 | Loss: 4.0646
Model saved!

Epoch 1 | Word 750488/750489 | Loss: 2.0959
Model saved!

Epoch 2 | Word 750488/750489 | Loss: 1.7565
Model saved!

Epoch 3 | Word 750488/750489 | Loss: 1.6047
Model saved!

Epoch 4 | Word 750488/750489 | Loss: 1.5146
Model saved!

Epoch 5 | Word 750488/750489 | Loss: 1.4516
Model saved!

Epoch 6 | Word 750488/750489 | Loss: 1.4007
Model saved!

Epoch 7 | Word 750488/750489 | Loss: 1.3591
Model saved!

Epoch 8 | Word 750488/750489 | Loss: 1.3279
Model saved!

Epoch 9 | Word 750488/750489 | Loss: 1.3466
Model saved!
```

The model achieves its best performance at **epoch 8**. Let's load that checkpoint:

```python
model.load()
```

```
Warning: embedding dimension does not match. Overriding embedded dimensions!
New dimensions: 100

Model loaded, resuming from epoch 8
```

### Results
```python
print(model.most_similar("cat"))
print(model.most_similar("train"))
print(model.most_similar("king"))
print(model.most_similar("love"))
print(model.most_similar("mouse"))
```

```aiignore
['rainfall', 'hereditary', 'snakes', 'implying', 'sps']
['albanian', 'albania', 'objectivism', 'pejorative', 'geography']
['british', 'alexander', 'john', 'six', 'seven']
['having', 'orbit', 'people', 'play', 'longer']
['levant', 'library', 'erik', 'lizard', 'cambodia']
```
These results **make some sense** but are not good results. The model should be trained more. 

There are a few ways we could improve the program: 
 - Filtering of common words such as "if, and, of, then, are, was..."
 - Lowering the learning rate or setting embedded dimensions to a higher value
 - Using more than 750k words
 - Better initialization of weights