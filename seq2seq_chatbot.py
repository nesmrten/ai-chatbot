import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

from utils.beam_search import beam_search
from utils.JsonDBEngine import JsonDbEngine
from utils.data_feeder import DataFeeder



class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(seq_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return nn.functional.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size + embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        attention_weights = self.attention(hidden, encoder_outputs)
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.decoder = nn.GRU(hidden_size * 2, hidden_size, num_layers=num_layers, dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = input.shape[0]
        max_length = target.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_length, vocab_size).to(self.device)
        encoder_output, hidden = self.encoder(input)

        # change shape of encoder_output from [batch_size, seq_len, hidden_size*num_directions] to [batch_size, hidden_size*num_directions, seq_len]
        encoder_output = encoder_output.permute(0, 2, 1)

        # create 3D tensor of encoder output for attention
        encoder_output = encoder_output.unsqueeze(2)

        decoder_input = target[:, 0]
        cell = torch.zeros_like(hidden)
        for t in range(1, max_length):
            embedded = self.dropout_layer(self.embedding(decoder_input))
            # change shape of embedded from [batch_size, embedding_size] to [batch_size, embedding_size, 1]
            embedded = embedded.unsqueeze(2)
            # apply attention
            attention_weights = torch.matmul(encoder_output, embedded).squeeze(2)
            attention_weights = nn.functional.softmax(attention_weights, dim=1)
            context = torch.matmul(encoder_output.permute(0, 2, 1), attention_weights.unsqueeze(2)).squeeze(2)

            rnn_input = torch.cat((embedded.squeeze(2), context), dim=1)
            output, hidden = self.decoder(rnn_input.unsqueeze(1), hidden)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1

        return outputs


class ChatBot:
    def __init__(self, model_path, db_path, data_path, max_input_len=20, beam_size=3, temperature=0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_engine = JsonDbEngine(db_path)
        self.data_feeder = DataFeeder(data_path, max_length=max_input_len)
        self.max_input_len = max_input_len
        self.beam_size = beam_size
        self.temperature = temperature

        model_dict = torch.load(model_path, map_location=self.device)
        encoder_params = model_dict['encoder_params']
        decoder_params = model_dict['decoder_params']
        self.input_vocab = model_dict['input_vocab']
        self.output_vocab = model_dict['output_vocab']

        self.encoder = Encoder(encoder_params['input_size'], encoder_params['embedding_size'],
                               encoder_params['hidden_size'], encoder_params['num_layers'], encoder_params['dropout'])
        self.attention = Attention(encoder_params['hidden_size'])
        self.decoder = Decoder(decoder_params['output_size'], decoder_params['embedding_size'],
                               decoder_params['hidden_size'], decoder_params['num_layers'], decoder_params['dropout'])
        self.seq2seq = Seq2Seq(self.encoder, self.attention, self.decoder, self.device)
        self.seq2seq.load_state_dict(model_dict['model_state_dict'])
        self.seq2seq.eval()

    def generate_response(self, query):
        input_seq = self.data_feeder.transform_input(query)
        input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(self.device)
        start_token = self.output_vocab['<sos>']
        end_token = self.output_vocab['<eos>']
        output_tokens = beam_search_decoder(self.seq2seq, input_tensor, start_token, end_token, self.beam_size,
                                             self.max_input_len, self.temperature)
        output_seq = [token.item() for token in output_tokens]
        output_seq = output_seq[1:-1]  # Remove start and end tokens
        output = self.data_feeder.transform_output(output_seq)
        return output

