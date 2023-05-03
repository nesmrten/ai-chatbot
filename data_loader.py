import os
import json
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader





class CornellMovieDialogsDataset(Dataset):
    def __init__(self, data_folder, max_length=20):
        self.max_length = max_length
        conversations_file = os.path.join(data_folder, 'conversations.json')
        self.conversations = self.load_conversations(conversations_file)
        self.questions, self.answers = self.preprocess_data(self.conversations)
        self.vocab = self.create_vocab(self.questions, self.answers)
        self.encoded_questions, self.encoded_answers = self.encode_data(self.questions, self.answers, self.vocab)
        self.padded_questions, self.padded_answers = self.pad_data(self.encoded_questions, self.encoded_answers, self.max_length)
    
    def __len__(self):
        return len(self.padded_questions)
    
    def __getitem__(self, idx):
        return self.padded_questions[idx], self.padded_answers[idx]
    
    def load_conversations(self, conversations_file):
        """
        Load conversation data from a JSON file.
        Returns a list of tuples containing the conversation pairs.
        """
        with open(conversations_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        pairs = []
        for conversation in conversations.values():
            lines = conversation['lines']
            for i in range(len(lines)-1):
                pair = (lines[i]['text'], lines[i+1]['text'])
                pairs.append(pair)

        return pairs
    
    def preprocess_data(self, pairs):
        """
        Preprocess the conversation pairs.
        Returns a tuple containing the questions and answers lists.
        """
        questions = []
        answers = []

        for pair in pairs:
            question = pair[0]
            answer = pair[1]

            # Remove non-alphabetic characters and lowercase the text
            question = re.sub(r'[^\w\s]', '', question).lower().strip()
            answer = re.sub(r'[^\w\s]', '', answer).lower().strip()

            # Tokenize the text
            question = question.split()
            answer = answer.split()

            # Add the question and answer to their respective lists
            questions.append(question)
            answers.append(answer)

        return questions, answers
    
    def create_vocab(self, questions, answers):
        """
        Create a vocabulary of words based on the questions and answers.
        Returns a dictionary of words and their corresponding indices.
        """
        word_counts = {}
        for question in questions:
            for word in question:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

        for answer in answers:
            for word in answer:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

        # Sort the words by frequency in descending order
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

        # Create the vocabulary dictionary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, word in enumerate(sorted_words):
            vocab[word] = i+2

        return vocab
    
    def encode_data(self, questions, answers, vocab):
      """
      Encode the questions and answers based on the vocabulary.
      Returns a tuple containing the encoded questions and answers.
      """
      encoded_questions = []
      encoded_answers = []

      for question in questions:
          encoded_question = [vocab.get(word, vocab['<UNK>']) for word in question]
          encoded_questions.append(encoded_question)

      for answer in answers:
          encoded_answer = [vocab.get(word, vocab['<UNK>']) for word in answer]
          encoded_answers.append(encoded_answer)

      return encoded_questions, encoded_answers
                                          
    def pad_data(self, encoded_questions, encoded_answers, max_length):
      """
      Pad questions and answers sequences to make them of equal length.
      """
      padded_questions = []
      padded_answers = []

      for question, answer in zip(encoded_questions, encoded_answers):
          if len(question) <= max_length and len(answer) <= max_length:
              padded_question = question + [self.vocab['<PAD>']] * (max_length - len(question))
              padded_answer = answer + [self.vocab['<PAD>']] * (max_length - len(answer))
              padded_questions.append(padded_question)
              padded_answers.append(padded_answer)

      return padded_questions, padded_answers
    
data_folder = "datasets/cornell_movie_dialogs_corpus/"
dataset = CornellMovieDialogsDataset(data_folder, max_length=20)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    padded_questions, padded_answers = batch
    print(padded_questions)
    print(padded_answers)

