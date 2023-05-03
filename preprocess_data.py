import re
import json
import os
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """
    This function takes a string of text and performs various cleaning operations on it,
    including removing punctuation and converting to lowercase.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()  # remove leading/trailing white space
    text = re.sub(' +', ' ', text)  # remove extra spaces
    return text

def tokenize(text):
    """
    This function takes a string of text and tokenizes it into a list of words.
    """
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    """
    This function removes stop words from a list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token in stop_words]
    return filtered_tokens

def preprocess(conversations_file_path, vocab_file_path, output_dir):
    # Load the conversations
    with open(conversations_file_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    # Extract the question-answer pairs and clean the text
    pairs = []
    for conversation in conversations.values():
        lines = conversation['lines']
        for i in range(len(lines) - 1):
            question = clean_text(lines[i]['text'])
            answer = clean_text(lines[i+1]['text'])
            pairs.append((question, answer))

    # Tokenize the text
    tokenized_pairs = []
    for pair in pairs:
        question_tokens = tokenize(pair[0])
        answer_tokens = tokenize(pair[1])
        tokenized_pairs.append((question_tokens, answer_tokens))

    # Remove stop words
    filtered_pairs = []
    for pair in tokenized_pairs:
        question_tokens = remove_stopwords(pair[0])
        answer_tokens = remove_stopwords(pair[1])
        filtered_pairs.append((question_tokens, answer_tokens))

    # Count the frequency of each word
    word_counts = Counter()
    for pair in filtered_pairs:
        for token in pair[0]:
            word_counts[token] += 1
        for token in pair[1]:
            word_counts[token] += 1

    # Create a vocabulary of words
    vocab = ['<pad>', '<unk>', '<start>', '<end>'] + [word for word, count in word_counts.items() if count >= 5]

    # Save the vocabulary to a file
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')

    # Encode the text data by mapping each word to its index in the vocabulary
    encoded_pairs = []
    for pair in filtered_pairs:
        question = [vocab.index(token) if token in vocab else 1 for token in pair[0]]
        answer = [vocab.index(token) if token in vocab else 1 for token in pair[1]]
        encoded_pairs.append((question, answer))

    # Pad the sequences to make them of equal length
    padded_questions = pad_data(encoded_questions, max_question_length)
    padded_answers = pad_data(encoded_answers, max_answer_length)

    # Split the data into training and validation sets
    questions_train, questions_val, answers_train, answers_val = train_test_split(padded_questions, padded_answers, test_size=0.1, random_state=42)

    # Save the preprocessed data
    data = {
        'vocab': vocab,
        'questions_train': questions_train.tolist(),
        'questions_val': questions_val.tolist(),
        'answers_train': answers_train.tolist(),
        'answers_val': answers_val.tolist()
    }

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Preprocessed data saved to {output_file_path}")

