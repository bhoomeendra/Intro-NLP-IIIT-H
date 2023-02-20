import torch
import torch.nn as nn

class LM(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embedding = self.
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
        self.LSTM = nn.LSTM(num_layers=self.config['num_layers'],input_size=self.config['embedding'],hidden_size=self.config['hidden'])
        self.convert_vocab = nn.Linear(in_features=self.config['hidden'],out_features=self.config['vocab_size'])
        self.softmax = nn.Softmax(dim=1)
    
    def get_embedding():
    	pass

    def forward(self,token_seq):
        x = self.embedding(token_seq)
        out,_ = self.LSTM(x) # (h_n,c_n)
        convert_output = self.convert_vocab(out)
        prob = self.softmax(convert_output)
        return prob
