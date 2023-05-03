import torch
import math

def beam_search(decoder, input_seq, init_hidden, max_length, beam_size):
    # Set up the initial input and hidden state for the decoder
    decoder_input = torch.tensor([[input_seq[0]]])
    decoder_hidden = init_hidden

    # Initialize the list of top K sequences and their log probabilities
    top_k_seqs = [{'sequence': [input_seq[0]], 'prob': 0.0}]
    complete_seqs = []

    # Run the decoder for each input token
    for i in range(1, len(input_seq)):
        # Create a new list to store the top K sequences for the current input token
        new_top_k = []

        # Expand each of the top K sequences with all possible next tokens
        for seq in top_k_seqs:
            seq_prob = seq['prob']
            seq_input = torch.tensor([[seq['sequence'][-1]]])
            seq_hidden = decoder_hidden

            decoder_output, decoder_hidden = decoder(seq_input, seq_hidden)
            decoder_output = decoder_output.squeeze()
            decoder_output = F.softmax(decoder_output, dim=0)

            # Get the top K tokens and their log probabilities
            top_k_probs, top_k_idxs = torch.topk(decoder_output, k=beam_size)

            # Add each of the top K sequences to the list of new top K sequences
            for j in range(beam_size):
                new_seq = {
                    'sequence': seq['sequence'] + [top_k_idxs[j].item()],
                    'prob': seq_prob + math.log(top_k_probs[j].item())
                }
                new_top_k.append(new_seq)

        # Sort the new top K sequences by their log probabilities
        new_top_k.sort(key=lambda x: x['prob'], reverse=True)

        # Keep only the top K sequences
        top_k_seqs = new_top_k[:beam_size]

    # Continue generating tokens until all the top K sequences have ended
    while len(complete_seqs) < beam_size:
        # Choose the top sequence from the list of top K sequences
        seq = top_k_seqs[0]
        seq_input = torch.tensor([[seq['sequence'][-1]]])
        seq_hidden = decoder_hidden

        decoder_output, decoder_hidden = decoder(seq_input, seq_hidden)
        decoder_output = decoder_output.squeeze()
        decoder_output = F.softmax(decoder_output, dim=0)

        # Get the top K tokens and their log probabilities
        top_k_probs, top_k_idxs = torch.topk(decoder_output, k=beam_size)

        # Add each of the top K sequences to the list of new top K sequences
        new_top_k = []
        for j in range(beam_size):
            new_seq = {
                'sequence': seq['sequence'] + [top_k_idxs[j].item()],
                'prob': seq['prob'] + math.log(top_k_probs[j].item())
            }
            new_top_k.append(new_seq)

        # Check if any of the new top K sequences have ended
        for seq in new_top_k:
            if seq['sequence'][-1] == 2: # 2 is the index of the end-of-sequence token
                complete_seqs.append(seq)
            else:
                top_k_seqs.append(seq)

        # Remove the current top sequence from the heap
        current_score, _, current_sequence = heappop(heap)

        # If the current sequence ends with the end token, add it to the completed sequences
        if current_sequence[-1] == end_token:
            completed_sequences.append((current_score, current_sequence))
            continue

        # Generate the next words in the sequence
        next_words = generate_next_words(current_sequence, model, device, vocab, beam_width)

        # Add the next sequences to the heap
        for next_word in next_words:
            score, _, sequence = current_score, current_sequence.copy(), current_sequence.copy()
            score += next_word[0]
            sequence.append(next_word[1])
            heappush(heap, (score, next_word[1], sequence))

        # Keep only the top-k sequences from the heap
        while len(heap) > beam_width:
            heappop(heap)

        # If there are no more sequences on the heap, stop the search
        if len(heap) == 0:
            break

        # Get the top sequence from the heap
        top_sequence = heappop(heap)[2]

        # Add the top sequence to the completed sequences
        completed_sequences.append((current_score, top_sequence))
