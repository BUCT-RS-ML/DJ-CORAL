import numpy as np
from sklearn import svm
from sklearn import metrics
import torch


def estimate_mu(_X1, _X2L, _X2U, _Y1, _Y2L,_Y2U):
    """
    Ref:
    J. Wang, W. Feng, Y. Chen, H. Yu, M. Huang, and P. S. Yu, “Visual domain adaptation with manifold embedded distribution alignment,” in
    Proceedings of the 26th ACM international conference on Multimedia,2018, pp. 402–410.
    """
    _X1=_X1.cpu().detach().numpy()
    _Y1=_Y1.cpu().detach().numpy()
    _X2=torch.cat((_X2L,_X2U),dim=0).cpu().detach().numpy()
    _Y2=torch.cat((_Y2L,_Y2U),dim=0).cpu().detach().numpy()
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(0, C):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c)  
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
         mu = 0
    return mu

def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int), np.ones(nb_target, dtype=int)))
    clf = svm.LinearSVC(random_state=1234)  #随机一半训练，一半测试
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist

