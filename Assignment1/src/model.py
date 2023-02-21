import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer
from Dataset import Dataset
## Embedding for SPECIAL Token 

class LM(nn.Module):
    
    def __init__(self,config,tokenizer):

        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.embedding_matrix = self.get_embedding()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.embedding_matrix).float())
        self.LSTM = nn.LSTM(num_layers=self.config['num_layers'],input_size=self.config['embedding'],hidden_size=self.config['hidden'])
        self.convert_vocab = nn.Linear(in_features=self.config['hidden'],out_features=self.config['vocab_size'])
        self.softmax = nn.Softmax(dim=1)
    

    def get_embedding(self):
        """This has to be aligned with the tokenizer ids"""

        vocab_embeddings = {}
        fi = open(self.tokenizer.pre_embedding_path,'rt')
        full_content = fi.read().strip().split('\n')
        for i in tqdm(range(len(full_content)),desc='Loading Glove'):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab_embeddings[i_word] = i_embeddings
        
        vocab_dim = len(vocab_embeddings['man'])
        unk_embedding = np.random.rand(vocab_dim) # Average of all tokens
        start_embedding = np.random.rand(vocab_dim)
        end_embedding = np.random.rand(vocab_dim)
        pad_embedding = np.zeros(vocab_dim)
        
        embedding = []
        
        glove_set = set(vocab_embeddings.keys())

        for i in tqdm(range(len(self.tokenizer.vocab)),desc='Embedding Matrix Formation:'):
            if self.tokenizer.idx_word[i] in glove_set:
                embedding.append(vocab_embeddings[self.tokenizer.idx_word[i]])
            elif self.tokenizer.idx_word[i] == '[PAD]':
                embedding.append(pad_embedding)
            elif self.tokenizer.idx_word[i] == '[UNK]':
                embedding.append(unk_embedding)
            elif self.tokenizer.idx_word[i] == '[STR]':
                embedding.append(start_embedding)
            elif self.tokenizer.idx_word[i] == '[END]':
                embedding.append(end_embedding)
        
        embedding = np.array(embedding)
        print(embedding.shape)
        embedding[1] = np.mean(embedding,axis=0) # mean of all tokens is UNK
        print(embedding.shape)
        return embedding


    def forward(self,token_seq):
        
        x = self.embedding(token_seq)
        out,_ = self.LSTM(x) # (h_n,c_n)
        convert_output = self.convert_vocab(out)
        prob = self.softmax(convert_output)
        return prob



if __name__=='__main__':
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    token = Tokenizer(doc_path='../data/Pride and Prejudice - Jane Austen.txt')
    data = Dataset(token,which='val')
    a,b = data[0]
    a = a.to(device)
    b = b.to(device)
    LM_config = {
    'epoch':10,
    'vocab_size': len(token.vocab),
    'embedding':50,
    'hidden':100,
    'lr':0.00001,
    'num_layers':1
    }
    LM_1 = LM(config=LM_config,tokenizer=token)
    LM_1.to(device)
    breakpoint()
