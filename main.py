from tokenizer import Tokenizer


if __name__ == "__main__":
    tk = Tokenizer("textFile.txt")
    print(tk.word2index)
    print(tk.index2word)
