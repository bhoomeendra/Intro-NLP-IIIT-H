import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def preplixity(ground_truth,predicted):
    # print("Ground truth , Predictions",ground_truth.shape,predicted.shape)
    index_gt = torch.argmax(ground_truth,dim=2)
    # print("Ground truth Index: ",index_gt)
    row = torch.arange(index_gt.shape[1]).to(device)
    probablity = predicted[0,row,index_gt]
    # print(probablity,probablity.shape)
    power_p = torch.pow(probablity,-1/probablity.shape[1])
    # print(power_p,power_p.shape)
    preplex = torch.prod(power_p)
    # print('\t',preplex.item())
    return preplex

def probablity(ground_truth,predicted):
    index_gt = torch.argmax(ground_truth,dim=1)
    # print("Ground truth Index: ",index_gt)
    row = torch.arange(index_gt.shape[0]).to(device)
    probablitys = predicted[row,index_gt]
    # print(probablity,probablity.shape)
    prob = torch.prod(probablitys)
    # print('\t',preplex.item())
    return prob
