import add_datamat as ad
import scipy.io as sio # read .mat files
from sklearn import preprocessing # Normalization data
import compute_miu as cm

import os
import time
import random
import datetime
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
#import data_loader
import numpy as np
import torch.nn as nn
from collections import defaultdict
from models import Prototypical
from loss import classification_loss_func, class_coral_alignment_loss_func, \
    get_prototype_label, CORAL
from utils import seed_everything

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Heterogeneous Domain Adaptation')

parser.add_argument('--cuda', type=str, default='0', help='Cuda index number')
parser.add_argument('--nepoch', type=int, default=300, help='Epoch amount') 
parser.add_argument('--partition', type=int, default=10, help='Number of partition') 
parser.add_argument('--prototype', type=str, default='two', choices=['two', 'three'],
                    help='how many prototypes used for domain and general alignment loss')
parser.add_argument('--layer', type=str, default='double', choices=['single', 'double'],
                    help='Structure of the projector network, single layer or double layers projector')
parser.add_argument('--d_common', type=int, default=256, help='Dimension of the common representation')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'mSGD', 'Adam'], help='optimizer options')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate') 

parser.add_argument('--temperature', type=float, default=4.0, help='source softmax temperature')
parser.add_argument('--alpha', type=float, default=0.1, help='Trade-off parameter of joint CORAL loss, set to 0 to turn off')


parser.add_argument('--combine_pred', type=str, default='Cosine',
                    choices=['Euclidean', 'Cosine', 'Euclidean_threshold', 'Cosine_threshold', 'None'],
                    help='the way of prototype predictions Euclidean, Cosine, None(not use)') #默认Cosine
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='All records save path')
parser.add_argument('--seed', type=int, default=1234, help='seed for everything')

args = parser.parse_args()
args.time_string = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')
savepath='result/'
name_str=str(args.alpha)+args.combine_pred
results_name = 'AIS-SAR_ResNet_Ts_'+str(args.temperature)+'_'+name_str

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if len(args.cuda) == 1:
        torch.cuda.set_device(int(args.cuda))

# seed for everything
seed_everything(args)


source_exp = [ad.AIS, ad.AIS, ad.AIS, ad.AIS, ad.AIS, ad.AIS, ad.AIS, ad.AIS, ad.AIS, ad.AIS]  #ad.SCS
target_exp = [ad.SAR1, ad.SAR2,ad.SAR3,ad.SAR4,ad.SAR5,ad.SAR6,ad.SAR7,ad.SAR8,ad.SAR9,ad.SAR10]
length = len(source_exp)

def test(model, configuration, srctar):
    model.eval()
    if srctar == 'source':
        loader = configuration['source_data']
        N = configuration['ns']
    elif srctar == 'labeled_target':
        loader = configuration['labeled_target_data']
        N = configuration['nl']
    elif srctar == 'unlabeled_target':
        loader = configuration['unlabeled_target_data']
        N = configuration['nu']
    else:
        raise Exception('Parameter srctar invalid! ')

    with torch.no_grad():
        feature, label = loader[0].float(), loader[1].reshape(-1, ).long()
        if torch.cuda.is_available():
            feature, label = feature.cuda(), label.cuda()
        classifier_output, _ = model(input_feature=feature)
        _, pred = torch.max(classifier_output.data, 1)
        n_correct = (pred == label).sum().item()
        acc = float(n_correct) / N * 100.

    return acc


def train(model, optimizer, configuration):  #model, model_d, optimizer, optimizer_d, configuration
    best_acc = -float('inf')
    list_miu = []
    # training
    for epoch in range(args.nepoch):

        start_time = time.time()
        model.train()
        
        optimizer.zero_grad()
        
        # prepare data
        source_data = configuration['source_data']
        l_target_data = configuration['labeled_target_data']
        u_target_data = configuration['unlabeled_target_data']
        source_feature, source_label = source_data[0].float(), source_data[1].reshape(-1, ).long()
        l_target_feature, l_target_label = l_target_data[0].float(), l_target_data[1].reshape(-1, ).long()
        u_target_feature = u_target_data[0].float()
        if torch.cuda.is_available():
            source_feature, source_label = source_feature.cuda(), source_label.cuda()
            l_target_feature, l_target_label = l_target_feature.cuda(), l_target_label.cuda()
            u_target_feature = u_target_feature.cuda()

        # forward propagation
        source_output, source_learned_feature = model(input_feature=source_feature)
        l_target_output, l_target_learned_feature = model(input_feature=l_target_feature)
        u_target_output, u_target_learned_feature = model(input_feature=u_target_feature)
        _, u_target_pseudo_label = torch.max(u_target_output, 1)  #f_ui
        if args.combine_pred == 'None':
            u_target_selected_feature = u_target_learned_feature
            u_target_selected_label = u_target_pseudo_label
            if epoch % 100 == 0:
                n_correct = (u_target_pseudo_label.cpu() == u_target_data[1].reshape(-1, ).long()).sum().item()
                acc_nn = float(n_correct) / configuration['nu'] * 100.
                print('Pesudo acc: (NN)', acc_nn)
        elif args.combine_pred.find('Euclidean') != -1 or args.combine_pred.find('Cosine') != -1:
            # get unlabeled data label via prototype prediction & network prediction s
            u_target_prototype_label = get_prototype_label(source_learned_features=source_learned_feature,
                                                           l_target_learned_features=l_target_learned_feature,
                                                           u_target_learned_features=u_target_learned_feature,
                                                           source_labels=source_label,
                                                           l_target_labels=l_target_label,
                                                           configuration=configuration,
                                                           combine_pred=args.combine_pred,
                                                           epoch=epoch)
            # select consistent examples
            u_target_selected_feature = u_target_learned_feature.index_select(dim=0, index=(
                    u_target_pseudo_label == u_target_prototype_label).nonzero().reshape(-1, ))   
            u_target_selected_label = u_target_pseudo_label.index_select(dim=0, index=(
                    u_target_pseudo_label == u_target_prototype_label).nonzero().reshape(-1, ))

            if epoch % 100 == 0:
                print('shared predictions:', len(u_target_selected_label), '/', len(u_target_pseudo_label))
                n_correct = (u_target_prototype_label.cpu() == u_target_data[1].reshape(-1, ).long()).sum().item()
                acc_pro = float(n_correct) / configuration['nu'] * 100.
                print('Prototype acc: (pro)', acc_pro)

        # ========================source data loss============================
        # labeled source data
        # CrossEntropy loss
        error_overall = classification_loss_func(source_output, source_label,ce_temperature=args.temperature)
        if epoch % 100 == 0:
            print('Use source CE loss: ', error_overall)
        # labeled target data
        
        target_CE_loss = classification_loss_func(l_target_output, l_target_label)
        error_overall += target_CE_loss 
        if epoch % 100 == 0:
            print('Use target CE loss: ',target_CE_loss)
        # ========================alignment loss============================
        # Calculate CORAL alignment loss
        if args.alpha:
            target_learned_feature=torch.cat((l_target_learned_feature, u_target_learned_feature), dim=0)
            domain_coral_alignment_loss = CORAL(source_learned_feature, target_learned_feature)

            u_target_selected_label = u_target_selected_label.reshape(-1, )
            class_alignment_loss = class_coral_alignment_loss_func(
                source_learned_features=source_learned_feature,
                l_target_learned_features=l_target_learned_feature,
                u_target_learned_features=u_target_selected_feature,
                source_labels=source_label,
                l_target_labels=l_target_label,
                u_target_pseudo_labels=u_target_selected_label,
                configuration=configuration,
                prototype=args.prototype)
            # estimate_miu
            mu = cm.estimate_mu(_X1=source_learned_feature,_X2L=l_target_learned_feature,_X2U=u_target_selected_feature,
                    _Y1=source_label,_Y2L=l_target_label,_Y2U=u_target_selected_label)
            # list_miu.append(mu)           
            joint_coral=(1-mu) * domain_coral_alignment_loss+ mu * class_alignment_loss
            if epoch % 10 == 0:
                print('compute miu is:', mu)    
            error_overall += args.alpha * joint_coral
            # general_align_list[epoch].append(general_alignment_loss.item())

            if epoch % 100 == 0:
                print('Use joint coral loss:', args.alpha * joint_coral)
        

        
        # backward propagation
        error_overall.backward()
        optimizer.step()


        # Testing Phase
        acc_src = test(model, configuration, 'source')
        acc_labeled_tar = test(model, configuration, 'labeled_target')
        acc_unlabeled_tar = test(model, configuration, 'unlabeled_target')
        end_time = time.time()
        if epoch % 10 == 0:
            print('ACC -> ', end='')
            print('Epoch: [{}/{}], {:.1f}s, Src acc: {:.4f}%, LTar acc: {:.4f}%, UTar acc: {:.4f}%'.format(
                epoch, args.nepoch, end_time - start_time, acc_src, acc_labeled_tar, acc_unlabeled_tar))

        if best_acc < acc_unlabeled_tar:
            best_acc = acc_unlabeled_tar
            
    # end for max_epoch
    print('Best Test Accuracy: {:.4f}%'.format(best_acc))
    return best_acc


if __name__ == '__main__':
    acc_net = np.zeros((args.partition,length)) 

    for j in range(0,length):
        print("Source domain: " + source_exp[j])
        print("Target domain: " + target_exp[j])
        source = sio.loadmat(source_exp[j])
        target = sio.loadmat(target_exp[j])

        xs = source['AIS_NGF11_data'] # read source data 
        xs_label = source['AIS_NGF11_labels'] - 1 # read source data labels, form 0 start
        xs = preprocessing.normalize(xs, norm='l2')

        xl = target['train_data']
        xl = preprocessing.normalize(xl, norm='l2')
        xl_label = target['train_label'] # read labeled target data labels, form 0 start
            
        xu = target['test_data'] # read unlabeled target data
        xu = preprocessing.normalize(xu, norm='l2')
        xu_label = target['test_label']  # read unlabeled target data labels, form 0 start

        ns, ds = xs.shape  # ns = number of source instances, ds = dimension of source instances
        nl, dt = xl.shape  # nl = number of labeled target instances, ds = dimension of all target instances
        nu, _ = xu.shape
        nt = nl + nu  # total amount of target instances
        class_number = len(np.unique(xs_label))  
        labeled_amount = 3
        # Generate dataset objects
        source_data = [torch.from_numpy(xs), torch.from_numpy(xs_label)]  
        labeled_target_data = [torch.from_numpy(xl), torch.from_numpy(xl_label)]
        unlabeled_target_data = [torch.from_numpy(xu), torch.from_numpy(xu_label)]
        configuration = {'ns': ns, 'nl': nl, 'nu': nu, 'nt': nt, 'class_number': class_number,
                     'labeled_amount': labeled_amount, 'd_source': ds, 'd_target': dt,
                     'source_data': source_data, 'labeled_target_data': labeled_target_data,
                     'unlabeled_target_data': unlabeled_target_data}
        
        result = 0.
        List_Miu = []
        for i in range(args.partition):
            #
            configuration = configuration
            model = Prototypical(configuration['d_source'], configuration['d_target'], args.d_common,
                             configuration['class_number'], args.layer)
           
            if torch.cuda.is_available():
                model = model.cuda()
                
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
                
            elif args.optimizer == 'mSGD':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                  weight_decay=0.001, nesterov=True) #weight_decay=0.001 is L2
                 
            elif args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.001)
                
            acc_net[i][j] = train(model, optimizer, configuration)
           
    np.savetxt(savepath+results_name+'.csv', acc_net, delimiter = ',')
    best_result_each=np.max(acc_net,axis=0)   
    mean_result_each=np.mean(acc_net,axis=0) 
    Aave=np.mean(best_result_each)
    Astd=np.std(best_result_each)

    Aave1=np.mean(mean_result_each)
    Astd1=np.std(mean_result_each)
    
    print('*'*20)
    param_str='alpha:'+str(args.alpha)+'//Ts: '+str(args.temperature)
    print(param_str)
    print('max_mean: %.2f''±''%.2f'%(Aave,Astd),end='    ')
    print('mean-mean:%.2f''±''%.2f'%(Aave1,Astd1))
    print('*'*20)       