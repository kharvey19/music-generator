import torch
import torch.nn as nn

class MelodyLSTM(nn.Module):
  # input_dim: number of possible chord types, hidden_dim: size of the internal representation, output_dim: number of possible note outputs
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, hidden_dim) # maps each chord index to a vector 
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim) # maps the LSTM's output to a note prediction 

  def forward(self, x):
    x = self.embedding(x)
    x, _ = self.lstm(x)
    return self.fc(x)


def load_model(input_dim, hidden_dim, output_dim, model_path):
    model = MelodyLSTM(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model