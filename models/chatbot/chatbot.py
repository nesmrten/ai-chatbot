import random
import json

class Chatbot:
    def __init__(self):
        with open('models/intents.json', 'r') as f:
            self.intents = json.load(f)
    def get_response(self, message):
        # TODO: Implement chatbot logic
        return "Hello, how can I help you?"