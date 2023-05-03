import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

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
    

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[:, -1, :]))
        return output, hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = input.shape[0]
        max_length = target.shape[1]
        vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(batch_size, max_length, vocab_size).to(self.device)
        encoder_output, hidden = self.encoder(input)
        
        decoder_input = target[:, 0]
        for t in range(1, max_length):
            output, hidden = self.decoder(decoder_input.unsqueeze(1), hidden)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
            
        return outputs
      # Define the Encoder network
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_length, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs shape: (batch_size, seq_length, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        return hidden, cell

# Define the Decoder network
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        x = x.unsqueeze(1)
        # x shape: (batch_size, 1, 1)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, 1, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # outputs shape: (batch_size, 1, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        predictions = self.fc(outputs.squeeze(1))
        # predictions shape: (batch_size, output_size)
        return predictions, hidden, cell

# Define the Seq2Seq network
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = input.shape[0]
        max_length = target.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_length, vocab_size).to(self.device)
        encoder_output, hidden = self.encoder(input)

        decoder_input = target[:, 0]
        cell = torch.zeros_like(hidden)
        for t in range(1, max_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1.squeeze(1)

