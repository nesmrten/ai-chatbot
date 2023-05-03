import os
import random
import torch
from torch.utils.data import Dataset
from utils.tokenizer import Tokenizer

from data_loader import load_cornell_movie_dialogs



class ChatbotDataset(Dataset):
    def __init__(self, data_dir, vocab_file, min_length, max_length, min_word_freq=5):

        self.data_dir = data_dir
        self.tokenizer = Tokenizer.load_from_file(vocab_file)
        self.input_file = os.path.join(data_dir, "input.txt")
        self.output_file = os.path.join(data_dir, "output.txt")
        self.input_data, self.output_data = self.load_data()

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_tokens = self.tokenizer.tokenize(self.input_data[index])
        output_tokens = self.tokenizer.tokenize(self.output_data[index])
        input_tensor = torch.LongTensor(input_tokens)
        output_tensor = torch.LongTensor(output_tokens)
        return input_tensor, output_tensor

    def load_data(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            input_data = f.readlines()
        with open(self.output_file, "r", encoding="utf-8") as f:
            output_data = f.readlines()
        return input_data, output_data

    def collate_fn(self, batch):
        input_data = []
        output_data = []
        input_lengths = []
        output_lengths = []

        for input_tensor, output_tensor in batch:
            input_data.append(input_tensor)
            output_data.append(output_tensor)
            input_lengths.append(len(input_tensor))
            output_lengths.append(len(output_tensor))

        max_input_len = max(input_lengths)
        max_output_len = max(output_lengths)

        padded_input_data = torch.zeros(len(batch), max_input_len).long()
        padded_output_data = torch.zeros(len(batch), max_output_len).long()

        for i, (input_tensor, output_tensor) in enumerate(batch):
            padded_input_data[i, :input_lengths[i]] = input_tensor
            padded_output_data[i, :output_lengths[i]] = output_tensor

        return padded_input_data, padded_output_data, input_lengths, output_lengths
