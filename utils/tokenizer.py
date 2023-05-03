import json
import os

class Tokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.num_words = 0

    def fit_on_texts(self, texts):
        """
        Updates the vocabulary based on a list of texts.

        Args:
            texts (list): A list of texts.
        """
        # Create a frequency dictionary of all the words in the texts
        word_freq = {}
        for text in texts:
            for word in text:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort the words by frequency and only keep the top n words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]

        # Assign an index to each word
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for idx, (word, freq) in enumerate(sorted_words):
            self.word2idx[word] = idx + 4

        # Create the inverse mapping from index to word
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def texts_to_sequences(self, texts):
        """
        Converts a list of texts to a list of sequences of token indices.

        Args:
            texts (list): A list of texts.

        Returns:
            list: A list of sequences of token indices.
        """
        sequences = []
        for text in texts:
            sequence = [self.word2idx.get(word, self.word2idx['<unk>']) for word in text]
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        """
        Converts a list of sequences of token indices to a list of texts.

        Args:
            sequences (list): A list of sequences of token indices.

        Returns:
            list: A list of texts.
        """
        texts = []
        for sequence in sequences:
            text = [self.idx2word.get(idx, '<unk>') for idx in sequence]
            texts.append(text)
        return texts

    def to_json(self):
        """Serializes the Tokenizer object to a JSON object."""
        return {'vocab_size': self.vocab_size, 'word2idx': self.word2idx}

    @classmethod
    def from_json(cls, json_dict):
        """Loads a Tokenizer object from a JSON object."""
        tokenizer = cls(vocab_size=json_dict['vocab_size'])
        tokenizer.word2idx = {token: int(idx) for token, idx in json_dict['word2idx'].items()}
        tokenizer.idx2word = {int(idx): token for token, idx in tokenizer.word2idx.items()}
        return tokenizer

    def save_vocab(self, file_path):
            """
            Saves the vocabulary to a file.

            Args:
                file_path (str): The path to the file.
            """
            with open(file_path, 'w') as f:
                json.dump(self.word2idx, f)

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            cls.word2idx = json.load(f)
        cls.idx2word = {idx: word for word, idx in cls.word2idx.items()}
