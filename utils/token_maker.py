from tokenizer import Tokenizer

# Instantiate a new Tokenizer object with a vocabulary size of 10000
tokenizer = Tokenizer(vocab_size=50000)

# Fit the tokenizer on a list of texts
texts = ['Hi, my name is Medea!', 'I came in love and peace.']
tokenizer.fit_on_texts(texts)

tokenizer.word2idx['<unk>'] = tokenizer.vocab_size
tokenizer.idx2word[tokenizer.vocab_size] = '<unk>'
tokenizer.vocab_size += 1

# Convert a list of texts to a list of token sequences
sequences = tokenizer.texts_to_sequences(texts)

# Convert a list of token sequences back to a list of texts
decoded_texts = tokenizer.sequences_to_texts(sequences)

# Save the tokenizer to a file
tokenizer.save_to_file('models/tokenizer.json')

# Load the tokenizer from a file
loaded_tokenizer = Tokenizer.load_from_file('models/tokenizer.json')
