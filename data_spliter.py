import os
import json
import argparse

def split_data(input_file, output_dir, num_chunks):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_size = len(data) // num_chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(data)
        chunk_data = [data[idx] for idx in range(start_idx, end_idx)]

        output_file = os.path.join(output_dir, f'conversations_{i}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)
        print(f'Saved chunk {i} to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split conversations data into multiple files.')
    parser.add_argument('--input_file', type=str, default='datasets/cornell_movie_dialogs_corpus/conversations.json',
                        help='Path to the input conversations file.')
    parser.add_argument('--output_dir', type=str, default='datasets/cornell_movie_dialogs_corpus/',
                        help='Directory to save the output chunk files.')
    parser.add_argument('--num_chunks', type=int, default=10, help='Number of chunks to split the data into.')
    args = parser.parse_args()

    split_data(args.input_file, args.output_dir, args.num_chunks)
