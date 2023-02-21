from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer
from Dataset import Dataset
from utils import preplixity
from model import LM
# "Use argparse to properly use the argumets"


def train_epoch(dataloader, model, loss_fn, optimizer,sch):
    size = len(dataloader.dataset)
    model.train()
    preplex = 0
    avg_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).to(torch.float32)
        pred = model(X)
        loss = loss_fn(pred, y)
        preplex += preplixity(y,pred)
        optimizer.zero_grad()
        # if batch%10 == 9:
        loss.backward()
        optimizer.step()
        # break
        avg_loss += loss.item()
        if batch % 300 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    preplex /= size
    avg_loss /= size
    print(f"\n Train Preplexity: {(preplex):>0.3f} \n loss: {avg_loss:>7f} ")
    return (preplex,avg_loss)


def validate_epoch(dataloader, model, loss_fn,which='Validation'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    preplex = 0
    with torch.no_grad():
        for X, y in dataloader:
            # token_sent(X)
            X, y = X.to(device), y.to(device).to(torch.float32)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            preplex += preplixity(y,pred)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    preplex /= size

    print(f"{which} Preplexity: \n : {(preplex):>0.5f}, Avg loss: {test_loss:>8f} \n")
    return (preplex,test_loss)


def train(LM_config):

	token = Tokenizer(doc_path='../data/Pride and Prejudice - Jane Austen.txt')
	train_data = Dataset(token,which='train')
	val_data = Dataset(token,which='val')
	train_dl =  DataLoader(train_data,shuffle=True)
	val_dl = DataLoader(val_data)

	LM_config['vocab_size'] = len(token.vocab)
	LM_1 = LM(config=LM_config,tokenizer=token)
	LM_1.to(device)
	loss = nn.CrossEntropyLoss()
	opt = torch.optim.Adam(params = LM_1.parameters(),lr=LM_config['lr'])
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt,max_lr=0.00001,steps_per_epoch=1,epochs=LM_config['epoch'])

	best_p = float('inf')
	epochs = LM_config['epoch']

	for t in range(epochs):

		print(f"Epoch {t+1}\n"+"-"*20)
		print(scheduler.get_last_lr()[0])
		tp,tl = train_epoch(train_dl, LM_1, loss,opt,scheduler)
		vp,vl = validate_epoch(val_dl, LM_1, loss)
		scheduler.step()

		if vp < best_p:
			best_p = vp
			torch.save(LM_1.state_dict(),'../data/LM_5.pth')
			print("model saved")
	print("Done!")

    
if __name__=='__main__':

    LM_config = {
    'epoch':10,
    'embedding':50,
    'hidden':100,
    'lr':0.000001,
    'num_layers':1
    }

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train(LM_config)
    print(device)
