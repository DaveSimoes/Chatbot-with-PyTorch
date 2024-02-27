import json
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    def __init__(self, intents_file):
        # Carregar dados do arquivo JSON
        with open(intents_file, 'r') as file:
            self.intents = json.load(file)['intents']

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        return self.intents[idx]
