import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChatbotDataset
from models.seq2seq import Seq2Seq
from utils.tokenizer import Tokenizer
from models.config import Config


def preprocess_data(data_dir, vocab_file):
    # Load tokenizer
    tokenizer = Tokenizer.load_from_file(vocab_file)

    # Load the dataset
    dataset = ChatbotDataset(data_dir, vocab_file)

    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    return tokenizer, dataloader


def train_model(model, tokenizer, dataloader):
    # Move the model to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    for epoch in range(Config.NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Move the inputs and targets to the device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, targets)

            # Compute the loss
            loss = criterion(outputs.view(-1, Config.OUTPUT_SIZE), targets.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % Config.PRINT_INTERVAL == Config.PRINT_INTERVAL - 1:
                print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / Config.PRINT_INTERVAL))
                running_loss = 0.0

    # Save the trained model
    torch.save(model.state_dict(), 'models/seq2seq_model.pth')


if __name__ == '__main__':
    # Preprocess the data
    tokenizer, dataloader = preprocess_data('data/cornell_movie-dialogs_corpus', 'data/cornell_movie-dialogs_corpus/vocab.txt')

    # Load the model
    model = Seq2Seq(tokenizer.vocab_size, Config.EMBEDDING_SIZE, Config.HIDDEN_SIZE, Config.OUTPUT_SIZE,
                    Config.NUM_LAYERS, dropout=0.5)

    # Train the model
    train_model(model, tokenizer, dataloader)
