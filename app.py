import os
import sys
from flask import Flask, render_template, request


# Add 'utils' directory to PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(project_root, 'utils')
sys.path.insert(0, utils_dir)

# Import modules from 'utils' directory
from data_feeder import DataFeeder
from code_feeder import CodeFeeder
from beam_search import BeamSearch


app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Chat page route
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        # Handle message sent by user
        message = request.form['message']
        # Process message using the chatbot model
        response = chatbot_model.process_message(message)
        # Return response to user
        return {'response': response}

    return render_template('chat.html')

# User settings page route
@app.route('/user_settings')
def user_settings():
    return render_template('user_settings.html')
