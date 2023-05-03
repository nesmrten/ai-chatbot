import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import CornellMovieDialogsDataset
from seq2seq_chatbot import Seq2Seq

data_folder = "datasets/cornell_movie_dialogs_corpus/"
max_length = 20
batch_size = 32

dataset = CornellMovieDialogsDataset(data_folder, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_size = output_size = len(dataset.vocab)

# Define the hyperparameters
learning_rate = 0.001
num_epochs = 10

# Initialize the model, loss function, and optimizer
input_size = output_size = len(dataset.vocab)
hidden_size = 256
num_layers = 2
dropout = 0.5

model = Seq2Seq(input_size, hidden_size, output_size, num_layers, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model using the dataloader
for epoch in range(num_epochs):
    for batch in dataloader:
        padded_questions, padded_answers = batch
        questions = padded_questions.to(device)
        answers = padded_answers.to(device)

        # Forward pass
        outputs = model(questions, answers)

        # Compute the loss
        loss = criterion(outputs.view(-1, output_size), answers.view(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model's performance on a validation set
validation_dataset = CornellMovieDialogsDataset(data_folder, max_length=20)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in validation_dataloader:
        padded_questions, padded_answers = batch
        questions = padded_questions.to(device)
        answers = padded_answers.to(device)

        # Forward pass
        outputs = model(questions, answers)

        # Compute the loss
        loss = criterion(outputs.view(-1, output_size), answers.view(-1))

        # Update the loss and token counts
        total_loss += loss.item() * padded_questions.size(0) * padded_questions.size(1)
        total_tokens += padded_questions.size(0) * padded_questions.size(1)

perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
print(f'Validation Perplexity: {perplexity:.4f}')

# Save the trained model to disk
model_path = "seq2seq_chatbot.pt"
torch.save(model.state_dict(), model_path)
