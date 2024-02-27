from src.chatbot_train import train_chatbot

intents_file = 'data/intents.json'
model_path = 'models/chatbot_model.pth'

train_chatbot(intents_file, model_path)
