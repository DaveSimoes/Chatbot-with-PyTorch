# chatbot_pytorch/src/chatbot_utils.py

import torch

def prepare_sequence(sequence, dataset):
    """
    Function to prepare a sequence of patterns to be used as input for the neural network.

    Parameters:
        sequence (list): List of patterns to be prepared.
        dataset (ChatbotDataset): Dataset object.

    Returns:
        torch.Tensor: Tensor containing the prepared sequence.
    """
    # Assume each pattern is a list of words
    # You can customize this based on the actual format of your data
    word_to_index = {word: idx for idx, word in enumerate(dataset.all_words)}

    # Convert the sequence of words into indices
    indices = [word_to_index[word] for word in sequence]

    # Create a PyTorch tensor from the indices
    tensor = torch.tensor(indices, dtype=torch.long)

    return tensor
