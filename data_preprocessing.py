import os
import json
import re

def load_lines(file_path):
    lines = {}
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = re.split(r'\s*\+\+\+\$\+\+\+\s*', line.strip())
            line_id = parts[0]
            speaker_id = parts[1]
            movie_id = parts[2]
            text = parts[4]
            lines[line_id] = {'speaker': speaker_id, 'text': text}
    return lines


def load_conversations(file_path, conversations_dict, lines_dict):
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            conv_parts = line.strip().split(' +++$+++ ')
            if len(conv_parts) == 4:
                conv_lines = json.loads(conv_parts[3].replace("'", '"'))
                conversations_dict[conv_parts[0]] = {
                    'character1_id': conv_parts[1],
                    'character2_id': conv_parts[2],
                    'lines': [lines_dict[line_id] for line_id in conv_lines],
                }

def extract_conversations_data(data_folder, output_file_path):
    lines_dict = {}
    conversations_dict = {}
    
    lines_file_path = os.path.join(data_folder, 'movie_lines.txt')
    conv_file_path = os.path.join(data_folder, 'movie_conversations.txt')
    
    lines_dict = load_lines(lines_file_path)
    load_conversations(conv_file_path, conversations_dict, lines_dict)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations_dict, f, ensure_ascii=False, indent=4)
    
    print(f"Extracted {len(conversations_dict)} conversations to {output_file_path}")

if __name__ == '__main__':
    data_folder = 'datasets/cornell_movie_dialogs_corpus'
    output_file_path = 'datasets/cornell_movie_dialogs_corpus/conversations.json'
    num_chunks = 10
    
    extract_conversations_data(data_folder, output_file_path)
