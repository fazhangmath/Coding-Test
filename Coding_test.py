import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt
import torch
import time

data0 = pd.read_csv('data.csv')

stock_list = list(data0.ticker.unique())
tmp = []
for i in range(len(stock_list)):
    if len(data0[data0['ticker']==stock_list[i]]) < 2005:
        tmp.append(stock_list[i])
for i in range(len(tmp)):
    stock_list.remove(tmp[i])

data = np.zeros([2005, 202, 2])
for i in range(len(stock_list)):
    data[:, i, :] = np.array(data0[data0['ticker']==stock_list[i]].iloc[:, 2:])
tmp = np.where(data==0)
for i in range(len(tmp[0])):
    data[tmp[0][i], tmp[1][i], tmp[2][i]] = data[tmp[0][i]-1, tmp[1][i], tmp[2][i]]
diff = np.diff(data, axis=0)
data = diff/data[:-1]


def portfolio(L1, L2):
    tmp = np.array([data[j:j + L1, :, :] for j in range(2005 - L1 - L2)])
    tmp = np.transpose(tmp, [0, 2, 1, 3])
    X = np.zeros([2005 - L1 - L2, 202, 2 * L1])
    X[:, :, 0:L1] = tmp[:, :, :, 0]
    X[:, :, L1:2 * L1] = tmp[:, :, :, 1]
    Y = np.array([np.sum(data[j + L1:j + L1 + L2, :, 0], axis=0) for j in range(2005 - L1 - L2)])

    Y_train = torch.Tensor(Y[0:1505 - L1 - L2].reshape(-1))
    Y_test = torch.Tensor(Y[1505 - L1 - L2:])
    X_train = torch.Tensor(X[0:1505 - L1 - L2].reshape(-1, X.shape[-1]))
    std = X_train.std([0])
    std[L1:] *= 4
    X_train = X_train / std
    X_test = torch.Tensor(X[1505 - L1 - L2:].reshape(-1, X.shape[-1]))
    X_test = X_test / std

    similarity = torch.cdist(X_test[:, 0:2 * L1], X_train[:, 0:2 * L1], p=1,
                             compute_mode='use_mm_for_euclid_dist_if_necessary')
    knn = similarity.topk(5000, largest=False)
    prediction = Y_train[knn.indices]
    alpha = prediction.mean(axis=1).reshape([500, 202])

    ret = np.zeros(int(500 / L2))
    weights = np.zeros([int(500 / L2), 202])
    for i in range(int(500 / L2)):
        place1 = torch.topk(alpha[L2 * i], k=20).indices
        place2 = torch.topk(-alpha[L2 * i], k=20).indices
        weights[i, place1] = 0.5 / len(place1)
        weights[i, place2] = -0.5 / len(place2)
        ret[i] = 0.5 * Y_test[L2 * i][place1].mean() - 0.5 * Y_test[L2 * i][place2].mean()

    plt.plot(np.cumsum(ret), label='Cumulative Return')
    plt.legend()
    plt.show()
    turnover = np.mean(np.abs(weights[1:] - weights[:-1]).sum(axis=1)) * 252 / L2
    ret_cum = np.array(np.cumprod(1 + ret))
    drawdown = np.maximum.accumulate(ret_cum)
    drawdown = (drawdown - ret_cum) / drawdown
    print('Annualized Turnover', turnover)
    print('Annualized Return', np.mean(ret) * 252 / L2)
    print('Annualized Risk', np.std(ret) * np.sqrt(252 / L2))
    print('Annualized Sharpe Ratio', np.mean(ret) / np.std(ret) * np.sqrt(252 / L2))
    print('Maximum Drawdown', np.max(drawdown))

    return None


portfolio(L1=10, L2=5)
portfolio(L1=10, L2=10)
portfolio(L1=10, L2=15)
portfolio(L1=10, L2=20)
