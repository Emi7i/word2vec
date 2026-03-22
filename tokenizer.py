import re


class Tokenizer:
    PATTERN = re.compile(r'[!"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~\t\n]')

    def __init__(self, text_file: str):
        self.text_file = text_file
        self.words = []

        self.train = []
        self.valid = []
        self.test = []

        self.word2index = {}
        self.index2word = {}

        self.load_words()
        self.tokenize(self.words)
        self.init_dataset()

    def init_dataset(self):
        num_words = len(self.words)
        train_end = int(num_words * 0.8)
        valid_end = int(num_words * 0.9)

        self.train = self.words[:train_end]
        self.valid = self.words[train_end:valid_end]
        self.test = self.words[valid_end:]

    def load_words(self):
        """ Loads words from a file, replaces specific characters and calls tokenize function """
        with open(self.text_file, "r", encoding="utf-8") as file:
            text = file.read().lower()
            text = self.PATTERN.sub(' ', text)
            self.words = text.split()

    def tokenize(self, list_of_words: list[str]):
        """ Maps the list of words into the index2word and word2index dictionaries """

        if len(list_of_words) == 0:
            print("No words to tokenize")
            return

        for word in list_of_words:
            if word not in self.word2index.keys():
                self.word2index[word] = len(self.word2index)
                self.index2word[len(self.index2word)] = word