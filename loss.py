import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def classification_loss_func(prediction, true_labels, ce_temperature=1.0):
    celoss_criterion = nn.CrossEntropyLoss()
    return celoss_criterion(prediction / ce_temperature, true_labels)


def class_coral_alignment_loss_func(source_learned_features, l_target_learned_features,
                                          u_target_learned_features, source_labels, l_target_labels,
                                          u_target_pseudo_labels, configuration, prototype):
    """
    class-level feature alignment: k-th class features of source, target, source-target,
    and calculate MSELOss between each pair
    :param prototype: how many prototypes used for general loss
    :param source_learned_features: source feature
    :param l_target_learned_features:  labeled target feature
    :param u_target_learned_features: unlabeled target feature
    :param source_labels: source groundtruth
    :param l_target_labels: label target groundtruth
    :param u_target_pseudo_labels: unlabeled target pseudo label
    :param configuration:
    :return:
    """
    class_number = configuration['class_number']
    mu_s = OrderedDict()  #创建字典对象
    mu_t = OrderedDict()

    if prototype == 'two':
        #
        for i in range(class_number):
            mu_s[i] = []
            mu_t[i] = []
        #
        assert source_learned_features.shape[0] == len(source_labels) #断言
        for i in range(source_learned_features.shape[0]):  
            mu_s[int(source_labels[i])].append(source_learned_features[i])

        assert l_target_learned_features.shape[0] == len(l_target_labels)
        for i in range(l_target_learned_features.shape[0]):
            mu_t[int(l_target_labels[i])].append(l_target_learned_features[i])

        assert u_target_learned_features.shape[0] == len(u_target_pseudo_labels)
        for i in range(u_target_learned_features.shape[0]):
            mu_t[int(u_target_pseudo_labels[i])].append(u_target_learned_features[i])

        error_general = 0

        for i in range(class_number):
            mu_s[i] = torch.stack(mu_s[i], 0).float()  #将张量纵向拼接
                      
            mu_t[i] = torch.stack(mu_t[i], 0).float()

            error_general += CORAL(mu_s[i], mu_t[i])

        return error_general


def get_prototype_label(source_learned_features, l_target_learned_features, u_target_learned_features, source_labels,
                        l_target_labels, configuration, combine_pred, epoch):
    """
    Ref: 
    S. Li, B. Xie, J. Wu, Y. Zhao, C. H. Liu, and Z. Ding, “Simultaneous semantic alignment network for heterogeneous domain adaptation,” in
    Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 3866-3874.

    get unlabeled target prototype label
    :param epoch: training epoch
    :param combine_pred: Euclidean, Cosine
    :param configuration: dataset configuration
    :param source_learned_features: source feature
    :param l_target_learned_features:  labeled target feature
    :param u_target_learned_features:  unlabeled target feature
    :param source_labels: source labels
    :param l_target_labels: labeled target labels
    :return: unlabeled target prototype label
    """
    def prototype_softmax(features, feature_centers):
        assert features.shape[1] == feature_centers.shape[1]
        n_samples = features.shape[0]
        C, dim = feature_centers.shape
        pred = torch.FloatTensor()
        for i in range(n_samples):
            if combine_pred.find('Euclidean') != -1:
                dis = -torch.pairwise_distance(features[i].expand(C, dim), feature_centers, p=2)
            elif combine_pred.find('Cosine') != -1:
                dis = torch.cosine_similarity(features[i].expand(C, dim), feature_centers)
            if not i:
                pred = dis.reshape(1, -1)
            else:
                pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)
        return pred

    assert source_learned_features.shape[1] == u_target_learned_features.shape[1]
    class_num = configuration['class_number']
    feature_dim = source_learned_features.shape[1]
    feature_centers = torch.zeros((class_num, feature_dim))
    for k in range(class_num):
        # calculate feature center of each class for source and target
        k_source_feature = source_learned_features.index_select(dim=0,
                                                                index=(source_labels == k).nonzero().reshape(-1, ))
        k_l_target_feature = l_target_learned_features.index_select(dim=0, index=(
                l_target_labels == k).nonzero().reshape(-1, ))
        feature_centers[k] = torch.mean(torch.cat((k_source_feature, k_l_target_feature), dim=0), dim=0) #原始的
    if torch.cuda.is_available():
        feature_centers = feature_centers.cuda()

    # assign 'pseudo label' by Euclidean distance or Cosine similarity between feature and prototype,
    # select the most confident samples in each pseudo class, not confident label=-1
    prototype_pred = prototype_softmax(u_target_learned_features, feature_centers)
    prototype_value, prototype_label = torch.max(prototype_pred.data, 1)

    # add threshold
    if combine_pred.find('threshold') != -1:
        #如果包含字符“threshold” 
        if combine_pred == 'Euclidean_threshold':
            # threshold for Euclidean distance
            select_threshold = 0.2
        elif combine_pred == 'Cosine_threshold':
            # Ref: Progressive Feature Alignment ddfor Unsupervised Domain Adaptation CVPR2019
            select_threshold = 1. / (1 + np.exp(-0.8 * (epoch + 1))) - 0.01
            # select_threshold = 0.1
        prototype_label[(prototype_value < select_threshold).nonzero()] = -1

    return prototype_label

def CORAL(source, target):
    """
    Ref:
    B. Sun and K. Saenko, “Deep coral: Correlation alignment for deep domain adaptation,” in European conference on computer vision. Springer,
    2016, pp. 443–450.
    """
    d = source.data.shape[1]
    ns = source.data.shape[0]
    nt =target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)/(ns-1)  #
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)/(nt-1)
    # frobenius norm  between source and target   
    loss = torch.sum(torch.mul((xc - xct), (xc - xct))) 
    # loss_sqrt =loss.sqrt()
    loss = loss/(4*d*d)
    return loss
