import torch

class Config:
    # General settings
    SEED = 42

    # Dataset settings
    DATASET_PATH = "data/cornell_movie-dialogs_corpus"
    VOCAB_PATH = "data/cornell_movie-dialogs_corpus/vocab.txt"

    # Model settings
    EMBEDDING_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    MAX_LENGTH = 50
    EOS_TOKEN = "<eos>"
    SOS_TOKEN = "<sos>"
    PAD_TOKEN = "<pad>"

    # Training settings
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    GRADIENT_CLIP = 5.0
    TEACHER_FORCING_RATIO = 0.5

    # Inference settings
    BEAM_SIZE = 5
    TEMPERATURE = 0.5

    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
