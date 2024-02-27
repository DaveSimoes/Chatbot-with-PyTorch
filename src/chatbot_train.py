import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import json
from chatbot_dataset import ChatbotDataset
from chatbot_model import ChatbotModel
from chatbot_utils import prepare_sequence

def train_chatbot(intents_file, model_path):
    # Define your hyperparameters here
    input_size = 100  # Input vector size (to be adjusted as required)
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = len(json.load(open(intents_file))['intents'])

    # Carregar dados
    dataset = ChatbotDataset(intents_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = ChatbotModel(input_size, hidden_size1, hidden_size2, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 100  # Adjust as necessary
    for epoch in range(num_epochs):
        for intent in dataloader:
            patterns = intent['patterns']
            tags = torch.tensor([dataset.intents.index(intent)], dtype=torch.long)

            # Prepare sequence (to be implemented in chatbot_utils.py)
            patterns_in = prepare_sequence(patterns, dataset)
            
            # Forward pass
            outputs = model(patterns_in)
            loss = criterion(outputs, tags)

            # Backward pass e ot
