import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.convert_to_json import convert_to_json
from seq2seq_chatbot import Encoder, Decoder, Seq2Seq
from utils.data_feeder import DataFeeder
from utils.JsonDBEngine import JsonDbEngine


def train(args):
    print("hey")
    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("Using device:", device)
    
    # Create the data loader
    data_feeder = DataFeeder(args.db_path)
    train_data = data_feeder.get_data("train")
    train_dataset = TextDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TextDataset.collate_fn
    )
    print("Data loaded")

    # Load the vocabulary
    with open(args.vocab_path, "r") as f:
        vocab = json.load(f)
    print("Vocabulary loaded")

    # Initialize the encoder and decoder
    encoder = Encoder(len(vocab), args.hidden_size, args.embedding_size).to(device)
    decoder = Decoder(len(vocab), args.hidden_size, args.embedding_size).to(device)
    print("Encoder and decoder initialized")

    # Initialize the model and optimizer
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Model and optimizer initialized")

    # Load the checkpoint if it exists
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {start_epoch-1}")
    else:
        start_epoch = 0

    # Train the model
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Starting epoch {epoch}")
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            input_seq = batch["input"].to(device)
            target_seq = batch["target"].to(device)
            output_seq = model(input_seq, target_seq)

            # Compute the loss
            loss = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])(output_seq.view(-1, len(vocab)), target_seq.view(-1))

            # Backward pass
            loss.backward()

            # Clip the gradients
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Update the parameters
            optimizer.step()

            # Add the loss to the total
            total_loss += loss.item()

            # Print the loss
            if batch_idx % args.print_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Batch {batch_idx}: Average loss: {avg_loss:.4f}")

        # Save the checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint-{epoch}.pt")
torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(train_loader)
       
        })