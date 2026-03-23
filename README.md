# word2vec
Implementation of Word2Vec (CBOW) from scratch using only NumPy, based on Mikolov et al. (2013). Includes tokenizer, negative sampling, and model checkpointing.

> NOTE: Check out explanation.md to get a better understanding of the code.

## Dataset
This project uses the text8 dataset. To download it:
1. Download from http://mattmahoney.net/dc/text8.zip
2. Extract the zip file
3. Place the `text8` file in the project root

For faster results, you can use the textFile that's in the repository.<br>
It contains the text of the book "His last bow : Some later reminiscences of Sherlock Holmes" by Arthur Conan Doyle.<br>
Link: https://www.gutenberg.org/ebooks/2350

## Results
```
most_similar("king")  → ['alexander', 'british', 'william']
most_similar("paris") → ['france', 'de', 'department']
most_similar("queen") → ['spain', 'daughter', 'governor']
```

## How to use the code
The only changes required should be in main.py where you can use functions and change hyperparameters.

### 1. Change the hyperparameters to your liking
What worked the best for me:
```python
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
```

> Detailed explanation of parameters in the steps below. 

### 2. Initialize the Tokenizer class:
```py
tk = Tokenizer(DATASET_PATH, FREQUENCY, MAX_VOCABULARY_SIZE)
```
- `DATASET_PATH` - The path to the dataset file (if the txt file is in root, just input the name of the file, like "text8")
- `FREQUENCY` - Threshold under which the words will be removed. If the words show up less than `FREQUENCY` times in the text, they will be removed to speed up the model.
- `MAX_VOCABULARY_SIZE` - The text8 dataset has 100M words. For faster computation, you can specify the max number of words taken from the text.

### 3. Initialize the Model class:
```py
model = Model(tk, EMBEDDING_DIM, WINDOW_SIZE)
```
- `EMBEDDING_DIM` - The dimensionality of the word vectors. The input and output matrices have the dimensions of **VOCABULARY_SIZE x EMBEDDING_DIM**.
- `WINDOW_SIZE` - The number of surrounding words taken before and after the target word when training the model to predict that word.

### 4. [OPTIONAL] Load the model
If you have already trained the model, you can load it with:
```py
model.load()
```
This will load the parameters and matrices from the `model` folder. If the loaded `EMBEDDING_DIM` does not match the current one, it will override it.

### 5. Train the model:
```py
model.train(EPOCHS, LEARNING_RATE, NEGATIVE_SAMPLES_NUM)
```
- `EPOCHS` - The number of epochs the model will train for. Note that when loading a model, it will continue from its last epoch and train until the `EPOCHS` variable is reached.
For example, if the model has already trained for 3 epochs, setting `EPOCHS` to 3 will not train the model further.
Setting `EPOCHS` to 6 will continue training for 3 more epochs.
- `LEARNING_RATE` - The learning rate for gradient descent.
- `NEGATIVE_SAMPLES_NUM` - Number of negative samples used during training. Set to 0 to disable negative sampling. Recommended values are between 5 and 20.

### 6. Find most similar words - validate the model:
```py
model.most_similar("king")
```
This uses cosine similarity on the learned embeddings to find the most semantically similar words to the input word. Returns a list of the top 5 most similar words.
> NOTE: Use it in a `print()` function. Example: `print(model.most_similar("king"))`.

## References
- [Word2vec - Wikipedia](https://en.wikipedia.org/wiki/Word2vec)
- Mikolov et al. (2013) - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Bengio et al. (2003) - [A Neural Probabilistic Language Model](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Dataset: [text8](http://mattmahoney.net/dc/text8.zip) by Matt Mahoney
