import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import DataLoader
import numpy as np
from seq2seq_chatbot import Seq2Seq
from chatbot import ChatBot
from dataset import ChatbotDataset
from torch.utils.data import DataLoader
from utils.JsonDBEngine import JsonDbEngine

# Initialize the database engine
db_engine = JsonDbEngine("chatbot_data/db")

chatbot = ChatBot()

train_dataset = ChatbotDataset(data_dir='datasets/cornell_movie_dialogs_corpus', vocab_file='vocab_file', 
                               min_length=5, max_length=20, min_word_freq=5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

input_size = output_size = len(train_dataset.vocab)

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
    for batch in train_loader:
        padded_questions, padded_answers = batch
        padded_questions = np.array([list(q) for q in padded_questions]).astype(np.int32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        padded_questions_tensor = torch.tensor(padded_questions, dtype=torch.int64).to(device)
        
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
    # Initialize the chatbot
    chatbot = Chatbot(db_engine)

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
