#Construction of ST-CGCN model
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
import time
import os
import datetime

# Matrix standardization
def normalize(A, symmetric=True):
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5).to(device))  # Degree matrix
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1).to(device))
        return D.mm(A)

# Adaptive data and dynamic adjacency matrix data loading
class SeqDataset(Dataset): 
    def __init__(self, dataf,AC_martix,AD_martix,data_wea, inputnum):
        self.imgseqs = dataf  # read in data
        self.num_samples = self.imgseqs.shape[1]  # Number of samples
        self.inputnum = inputnum  # Length of spatial sequence
        self.inputnumT = 12*self.inputnum  # Length of temporal sequence
        self.AC_M = AC_martix
        self.AD_M = AD_martix
        self.AW_data= data_wea
    def __getitem__(self, index):
        current_index = np.random.choice(range(self.inputnumT, self.num_samples))
        current_label = self.imgseqs[:,current_index]
        current_imgs=self.imgseqs[:,current_index-self.inputnum:current_index]
        current_imgs=torch.FloatTensor(current_imgs)
        current_imgs1 = []
        for i in range(current_index - self.inputnumT, current_index, 12):
            img = self.imgseqs[:,i].T
            current_imgs1.append(list(img))
        current_imgs1=torch.FloatTensor(current_imgs1).T
        current_label=torch.FloatTensor(current_label)
        AW_M=np.ones((Nodes_num,Nodes_num))*self.AW_data[current_index]
        AW_M[np.eye(Nodes_num,dtype=np.bool)]=0
        AW_M=normalize(torch.FloatTensor(AW_M).to(device), True)
        return current_imgs,current_imgs1, current_label, self.AC_M,self.AD_M,AW_M
    def __len__(self):
        return self.imgseqs.shape[1]

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):

        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


#STCGCN-bone
class STCGCN_Net(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(STCGCN_Net, self).__init__()
        self.outlen = dim_out
        self.inlen = dim_in
        self.fc1 = nn.Linear(Nodes_num,Nodes_num, bias=True)
        self.fc2 = nn.Linear(Nodes_num,Nodes_num, bias=True)
        self.fc3 = nn.Linear(Nodes_num,Nodes_num, bias=True)
        self.Fus1 = nn.Sequential(
            Conv(3, 1, 1, 1),
        )
        self.fc4 = nn.Linear(8, 4, bias=True)
        self.fc5 = nn.Linear(4, 1, bias=True)
        #spatial
        self.spatial=net1 #Replace with your spatial feature extraction module
        #temporal
        self.temporal=net2 #Replace with your temporal feature extraction module
        self.fc6 = nn.Linear(Nodes_num, Nodes_num, bias=True)
        self.fc7 = nn.Linear(Nodes_num, Nodes_num, bias=True)

    def forward(self,X,X1,AC,AD,AW):
        Batch=X.shape[0]
        X0=X
        #spatial
        AC = self.fc1(AC).view(Batch,-1,Nodes_num,Nodes_num)
        AD = self.fc2(AD).view(Batch,-1,Nodes_num,Nodes_num)
        AW = self.fc3(AW).view(Batch,-1,Nodes_num,Nodes_num)
        AF=self.Fus1(torch.cat((AC,AD,AW),1)).view(-1,Nodes_num,Nodes_num) #矩阵融合
        #spatial block
        XS=self.spatial(AF,X) #Replace with your spatial feature extraction module
        #temporal block
        XT=self.temporal(X1) #Replace with your temporal feature extraction module
        #Fuse
        out=torch.tanh(self.fc6(XS)+self.fc7(XT))
        return out

#model training
def train(epoch):
    erro=[]
    epoch_start_time=time.time()
    for batch_idx, (batch_x,batch_x1, batch_y,batch_ac,batch_ad,batch_aw) in enumerate(train_loader, 0):
        inputs,inputs1, label, AC, AD, AW = Variable(batch_x).to(device),Variable(batch_x1).to(device), Variable(batch_y).to(device), \
                                    Variable(batch_ac).to(device), Variable(batch_ad).to(device), Variable(batch_aw).to(device)
        output = model(inputs,inputs1,AC, AD, AW)
        output=output.view(-1,Nodes_num,OUT_SIZE)
        label=label.view(-1,Nodes_num,OUT_SIZE)
        loss_func = nn.MSELoss()
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        erro.append(loss.data.cpu().numpy())
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print('Train Epoch:[{:03d}/{:03d}] Sec: {:.3f} Loss: {:.6f}'.format(
        epoch, num_epoch, time.time() - epoch_start_time, np.mean(erro)))
    loss_list.append(np.mean(erro))
    if epoch % 10 == 0:
        torch.save(model, model_path + str(epoch) + ".pth") #模型保存

#Model test verification
def model_test(item,AC_martix, AD_martix):
    test_label=torch.FloatTensor(data_test[:,item]).to(device)
    current_imgs = torch.FloatTensor(data_test[:, item - SEQ_SIZE:item]).to(device)
    current_imgs1 = []
    for i in range(item - 12*SEQ_SIZE, item, 12):
        img = data_test[:, i].T
        current_imgs1.append(list(img))
    current_imgs1 = torch.FloatTensor(current_imgs1).T.to(device)
    AW_M = np.ones((Nodes_num, Nodes_num)) * data_wea[item]
    AW_M[np.eye(Nodes_num, dtype=np.bool)] = 0
    AW_M = normalize(torch.FloatTensor(AW_M).to(device), True)
    current_imgs=current_imgs.view(1,Nodes_num,SEQ_SIZE)
    current_imgs1=current_imgs1.view(1,Nodes_num,SEQ_SIZE)
    test_label=test_label.view(1,Nodes_num,OUT_SIZE)
    output = model(current_imgs, current_imgs1, AC_martix, AD_martix, AW_M)
    out_P = output * data_std + data_mean
    out_T = test_label * data_std + data_mean
    out_P=out_P.view(-1)
    out_T = out_T.view(-1)
    return out_P.data.cpu().numpy(), out_T.data.cpu().numpy()

if __name__=='__main__':
    #Model Initialization
    BATCH_SIZE = 10
    SEQ_SIZE = 8  # Input Dimension
    OUT_SIZE = 1  #Output Dimension
    learning_rate = 0.0001  # Learning rate
    num_epoch=500 #Training times
    timeid = time.time()
    timeArray = time.localtime(timeid)
    timestart = time.strftime("%Y%m%d %H-%M-%S", timeArray) #time record
    model_kind='stcgcn'  #model name
    data_kind=['PEMS03','PEMS04','PEMS07','PEMS08','SZ_taxi'] #datasets
    data_index=0 #id
    data_id=0 #id of data kind
    #build
    model_path='model_save/'+model_kind+'-'+data_kind[data_index]+'['+str(data_id)+']-'+str(timestart)+'/' #Model storage folder
    data_path = '../data_save/' +model_kind+'-'+data_kind[data_index]+'['+str(data_id)+']-'+str(timestart)+'/'  # Output data storage folder
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU
    AC_martix=pd.read_csv('../data/data_martix/'+data_kind[data_index]+'['+str(data_id)+']'+'/data_corr.csv',header=0,index_col=0)
    AC_martix=AC_martix.values
    AD_martix=pd.read_csv('../data/data_martix/'+data_kind[data_index]+'['+str(data_id)+']'+'/data_dist.csv',header=0,index_col=0)
    AD_martix=AD_martix.values
    AD_martix0=AD_martix
    AD_martix=normalize(torch.FloatTensor(AD_martix0).to(device), True) #Regularization
    AC_martix=normalize(torch.FloatTensor(AC_martix).to(device), True)
    data_wea=pd.read_csv('../data/data_martix/'+data_kind[data_index]+'['+str(data_id)+']'+'/data_wea.csv',header=0,index_col=0).values
    #load datasets
    file = np.load('../data/source_data/'+str(data_kind[data_index])+'/'+str(data_kind[data_index])+'.npz', allow_pickle=True)
    data1 = file['data'].T
    data = pd.DataFrame(data1[data_id].T)
    data = data.values[:25920, :].T
    print("input_size:"+ str(data.shape))
    #Data standardization
    data_mean = data.mean()
    data_std = data.std()
    data_B = (data - data_mean) / data_std  # Standardization
    #Dataset partitioning 7:3
    day_len=int(data.shape[1]/288)
    train_len=int(day_len*0.7)
    val_len=day_len-train_len
    print("train_days:%d,test_days:%d" % (train_len,val_len))
    data_train = data_B[:,0:train_len*288]
    print("train_data_size:"+ str(data_train.shape))
    data_test = data_B[:,train_len*288 - SEQ_SIZE:day_len*288]
    print("test_data_size:" + str(data_test.shape))
    Nodes_num=data_train.shape[0]
    #Load Training Set
    train_data = SeqDataset(data_train,AC_martix,AD_martix,data_wea, inputnum=SEQ_SIZE)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    #Load Model
    model = STCGCN_Net(SEQ_SIZE, OUT_SIZE)
    model = model.to(device)
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    print("----model--training---- ")
    for epoch in range(num_epoch):
        train(epoch)
    print("----model--train---End---- ")
    #Draw loss curve
    plt.plot(loss_list, '.-')
    plt.xlabel('times')
    plt.ylabel('Test loss')
    data_loss = pd.DataFrame(loss_list)
    data_loss.to_csv(data_path+ 'loss_data.csv')
    plt.savefig(data_path + 'loss_fig_.svg', format='svg')
    # plt.show()
    #Model validation test
    data_P = []
    data_T = []
    for i in range(12*SEQ_SIZE, data_test.shape[1]):
        P,T=model_test(i,AC_martix,AD_martix)
        data_P.append(P)
        data_T.append(T)
    dataf1 = pd.DataFrame(data_T)
    dataf1.to_csv(data_path+"data_true.csv")
    dataf1 = pd.DataFrame(data_P)
    dataf1.to_csv(data_path+"data_predict.csv")









