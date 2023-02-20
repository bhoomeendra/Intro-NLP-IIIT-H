import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import pdb

def tokenizer(text,idx_word,word_idx,final_vocab):
    tokens = []
    text.lower()
    text = text.strip('\n')
    text = idx_word[2]+' '+text+' '+idx_word[3]
    # print(text)
    for word in text.split():
        if word in final_vocab:
            tokens.append(word_idx[word])
        else:
            tokens.append(word_idx['[UNK]']) # update []
    return tokens



class Dataset(Dataset):
    def __init__(self,data,vocab_size):
        self.data = data
        self.vocab_size = vocab_size
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        # print(self.data[idx])
        out = tokenizer(self.data[idx])
        inputs = out[:-1]# Last token is [END] will not include in data
        labels = out[1:]# First token is [STR] will never be predected
        # print(out)
        return torch.tensor(inputs),torch.nn.functional.one_hot(torch.tensor(labels),num_classes=self.vocab_size)




class LM(nn.Module):
    
    def __init__(self,config,embedding):
        super().__init__()
        self.config = config
        # This will tak token ids an make enbedding out of it
        # Embedding are from glove and should be frezzed from learning
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding).float())
        self.LSTM = nn.LSTM(num_layers=self.config['num_layers'],input_size=self.config['embedding'],hidden_size=self.config['hidden'])
        self.convert_vocab = nn.Linear(in_features=self.config['hidden'],out_features=self.config['vocab_size'])
        self.softmax = nn.Softmax(dim=1)
    def forward(self,token_seq):
        x = self.embedding(token_seq)
        out,_ = self.LSTM(x) # (h_n,c_n)
        convert_output = self.convert_vocab(out)
        prob = self.softmax(convert_output)
        return prob

def sent_probablity(ground_truth,pred):
    # print("Ground truth , Predictions",ground_truth.shape,pred.shape)
    index_gt = torch.argmax(ground_truth,dim=1)
    
    # print("Ground truth Index: ",index_gt)
    # pdb.set_trace()
    row = torch.arange(index_gt.shape[0]).to(device)
    probablity = pred[row,index_gt]
    print(probablity,probablity.shape)
    # print(power_p,power_p.shape)
    prob = torch.prod(probablity)
    return prob

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

if __name__=='__main__':
    file = open('../data/LM_5_data.pkl','rb')
    config = pickle.load(file)
    final_vocab = config['final_vocab']
    word_idx = config['word_idx']
    idx_word = config['idx_word']
    embedding = config['embedding']
    LM_config = config['LM_config']
    #test_data = config['test_data'] # list of strings
    checkpoint = torch.load('../data/LM_5.pth')
    LM_1 = LM(LM_config,embedding).to(device)
    LM_1.load_state_dict(checkpoint)
    LM_1.eval()
    while(1):
        sent = input("input sentence: ").lower()
        tokens = tokenizer(sent,idx_word,word_idx,final_vocab)
        tok_sent = torch.tensor(tokens[:-1]).to(device)
        ground_truth = torch.nn.functional.one_hot(torch.tensor(tokens[1:]),num_classes=len(final_vocab)).to(device)
        pred = LM_1(tok_sent)
        print(sent_probablity(ground_truth,pred))

