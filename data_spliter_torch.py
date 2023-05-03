import argparse
import json
import os
import torch


def split_data(input_file, output_dir, num_chunks):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert the dictionary into a list of conversations
    conversations = []
    for conversation_id, conversation in data.items():
        conversations.append(conversation)
    
    chunk_size = len(conversations) // num_chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(conversations)
        chunk_data = conversations[start_idx:end_idx]

        output_file = os.path.join(output_dir, f'conversations_{i}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)
        print(f'Saved chunk {i} to {output_file}')

    num_conversations = len(conversations)
    chunk_size = num_conversations // num_chunks

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else num_conversations
        chunk_data = conversations[start_idx:end_idx]

        chunk_file_path = os.path.join(output_dir, f'chunk_{i+1}.json')
        with open(chunk_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)

        print(f'Saved {len(chunk_data)} conversations to {chunk_file_path}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='datasets/cornell_movie_dialogs_corpus/conversations.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='datasets/cornell_movie_dialogs_corpus/processed')
    parser.add_argument('--num_chunks', type=int, required=True,
                        help='10')
    args = parser.parse_args()

    split_data(args.input_file, args.output_dir, args.num_chunks)


if __name__ == '__main__':
    main()
