import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataFeeder:
    def __init__(self, db_engine):
        self.db_engine = db_engine

    def clean_text(self, text):
        # remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        # remove punctuation and lowercase text
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if not word in stop_words]
        return ' '.join(filtered_text)

    def add_text_to_dataset(self, text, intent_name):
        cleaned_text = self.clean_text(text)
        document = {'text': cleaned_text, 'intent': intent_name}
        self.db_engine.insert('chatbot_data', document)

    def add_texts_to_dataset(self, texts, intent_name):
        for text in texts:
            self.add_text_to_dataset(text, intent_name)
