import pandas
import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam

from tqdm import tqdm

class net(nn.Module):
    def __init__(self, size):
        super(net, self).__init__()
        
        self.l1 = nn.Linear(size, 1500)
        self.l2 = nn.Linear(1500, 1000)
        self.l3 = nn.Linear(1000, 500)
        self.l4 = nn.Linear(500, 250)
        self.l5 = nn.Linear(250, 250)
        self.l6 = nn.Linear(250, 1)
        
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(self.act(x))
        x = self.l3(self.act(x))
        x = self.l4(self.act(x))
        x = self.l5(self.act(x))
        x = self.l6(self.act(x))
        
        return x

def separate(pd_arr, column_name):
    y = torch.Tensor(np.array(pd_arr[column_name]))
    x = torch.Tensor(np.array(pd_arr.drop([column_name], 1)))
    
    return x, y

def read(file_name):
    data_file = pandas.read_stata(file_name+".dta")
    return data_file

def process_stada(data_file): 
    
    data_file.drop(['intmonth', 'lineno', 'hurespli', 'hrlonglk', 'serial', 'hhnum', 'qstnum', 'occurnum', 'ym', 'ym_file', 'weight', 'earnwtp', 'minsamp', 'hrsample', 'earnwt', ], 1, inplace=True)
    df = data_file.hhid.value_counts()
    data_file = data_file[data_file.hhid.isin(df.index[df.lt(5)])]
    data_file.drop(['hhid'], 1, inplace=True)
    data_file = data_file[data_file['smsastat'] == 'Metropolitan']
    
    
    data_file.drop(['smsastat', 'centcity', 'icntcity', 'msafips', 'cmsacode', 'county', 'icntcity', ], 1, inplace=True)
    
    data_file = data_file[~data_file.smsa93.isna()]
    data_file = data_file[~data_file.earnwke.isna()]
    size = data_file.smsa93
    data_file.drop(['smsa93'], 1, inplace=True)
    
    data_ext = pandas.get_dummies(data_file, dummy_na = True)
    data_file = data_ext.fillna(value = -1)
    data_file = pandas.concat([data_file, size], axis = 1)
    
    small_list = ['100,000 - 249,999', '250,000 - 499,999', '500,000 - 999,999']
    data_metropolitan_big = data_file[~data_file.smsa93.isin(small_list)]
    data_non = data_file[data_file.smsa93.isin(small_list)]
    data_metropolitan_big.drop(['smsa93'], 1, inplace=True)
    data_non.drop(['smsa93'], 1, inplace=True)
    
    x_1, y_1 = separate(data_metropolitan_big, 'earnwke')
    x_2, y_2 = separate(data_non, 'earnwke')
    
    return x_1, y_1, x_2, y_2

def stada_to_panda():

    file_lis = ['morg01', 'morg02']
    
    df = read(file_lis[0])
    
    for i in file_lis[1:]:
        df = df.append(read(i))
    
    x_1, y_1, x_2, y_2 = process_stada(df)
    
    return x_1, y_1, x_2, y_2

x_1, y_1, x_2, y_2 = stada_to_panda()
    
first_layer_size = x_1.shape[1]

net_1 = net(first_layer_size)
net_2 = net(first_layer_size)

opt1 = Adam(net_1.parameters(), lr = 0.0005)
opt2 = Adam(net_2.parameters(), lr = 0.0005)

loss_func = nn.MSELoss()

for i in tqdm(range(1)):
    y_hat = net_1(x_1).squeeze()
    
    loss = loss_func(y_hat, y_1)
    loss.backward()
    opt1.step()
    net_1.zero_grad()
    opt1.zero_grad()
    
    y_hat = net_2(x_2).squeeze()
    
    loss = loss_func(y_hat, y_2)
    loss.backward()
    opt2.step()
    net_2.zero_grad()
    opt2.zero_grad()


y_hat_1 = net_1(x_1)
y_hat_2 = net_2(x_1)

print(torch.mean(y_hat_1- y_hat_2))

similarity_metrix = torch.cdist(x_1, x_2)

y_match = torch.zeros_like(y_hat_1)

for i in range(y_match.shape[1]):
    argmin = torch.argmin(similarity_metrix[i])
    y_match = y_1[i] - y_2[argmin]

print(torch.mean(y_match))
