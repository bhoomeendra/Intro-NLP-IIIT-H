from tokenizer import Tokenizer
from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm

class Dataset(Dataset):

    def __init__(self,tokenizer,which:str,random_seed:int=42):
    
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.vocab)
        random.seed(42)
        data = self.tokenizer.sentence_maker(open(self.tokenizer.doc_path).read())
        random.shuffle(data)
        train = int(len(data)*0.7)
        val = int(len(data)*0.15)
        if which == 'train':
        	self.data = [self.tokenizer.sent2token(x) for x in tqdm(data[:train],desc=f'{which} loading') if self.tokenizer.sent2token(x) is not None ]
        elif which == 'val':
        	self.data = [self.tokenizer.sent2token(x) for x in tqdm(data[train:train+val],desc=f'{which} loading') if self.tokenizer.sent2token(x) is not None]
        elif which == 'test':
        	self.data = [self.tokenizer.sent2token(x) for x in tqdm(data[train+val:],desc=f'{which} loading') if self.tokenizer.sent2token(x) is not None]
    
    
    def __len__(self):
    
        return len(self.data)
    

    def __getitem__(self,idx):
    
        out = self.data[idx]
        # print("This is error: ",out)
        inputs = out[:-1]# Last token is [END] will not include in data
        labels = out[1:]# First token is [STR] will never be predected
        return torch.tensor(inputs),torch.nn.functional.one_hot(torch.tensor(labels),num_classes=self.vocab_size)



if __name__ == '__main__':


    token = Tokenizer(doc_path='../data/Pride and Prejudice - Jane Austen.txt')

    print(token.sent2token("This is a okay sentence"))
    print(token.token2sent([324,34,4,76,78,1,2,0,3]))

    train_data = Dataset(token,which='train')
    val_data = Dataset(token,which='val')
    test_data = Dataset(token,which='test')
    
    print("************************* Train Start:******************************************")

    for i in range(len(train_data)):
        print(i,len(train_data[i][0]),end=' ')
    # print(train_data[691])
    # breakpoint()
    print("************************* Validation Start:******************************************")

    
    for i in range(len(test_data)):
        print(i,len(test_data[i][0]),end=' ')

    print("************************* Test Start:******************************************")
    
    for i in range(len(val_data)):
        print(i,len(val_data[i][0]),end=' ')	

    breakpoint()