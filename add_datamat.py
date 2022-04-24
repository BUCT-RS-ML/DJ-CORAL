import scipy.io as sio # read .mat files
import numpy as np
'''
load data from the /data/ dir
'''

AIS = r'data/AIS_NGF11_data.mat'


OR=r'data/SAR_ResNet'
SAR1 = OR + '/sar_resnet18_1.mat' 
SAR2 = OR + '/sar_resnet18_2.mat'  
SAR3 = OR + '/sar_resnet18_3.mat'
SAR4 = OR + '/sar_resnet18_4.mat'
SAR5 = OR + '/sar_resnet18_5.mat'
SAR6 = OR + '/sar_resnet18_6.mat'
SAR7 = OR + '/sar_resnet18_7.mat'
SAR8 = OR + '/sar_resnet18_8.mat'
SAR9 = OR + '/sar_resnet18_9.mat'
SAR10 = OR +'/sar_resnet18_10.mat'


## Decaf6
# OR=r'data/SAR_Decaf6'
# SAR1 = OR + '/JKF_sar_decaf6_1.mat'  #
# SAR2 = OR + '/JKF_sar_decaf6_2.mat'
# SAR3 = OR + '/JKF_sar_decaf6_3.mat'
# SAR4 = OR + '/JKF_sar_decaf6_4.mat'
# SAR5 = OR + '/JKF_sar_decaf6_5.mat'
# SAR6 = OR + '/JKF_sar_decaf6_6.mat'
# SAR7 = OR + '/JKF_sar_decaf6_7.mat'
# SAR8 = OR + '/JKF_sar_decaf6_8.mat'
# SAR9 = OR + '/JKF_sar_decaf6_9.mat'
# SAR10 = OR +'/JKF_sar_decaf6_10.mat'
## SURF
# OR="data/SAR_SURF"
# SAR1 = OR + "/JKF_sar_surf_1.mat"
# SAR2 = OR + '/JKF_sar_surf_2.mat'
# SAR3 = OR + '/JKF_sar_surf_3.mat'
# SAR4 = OR + '/JKF_sar_surf_4.mat'
# SAR5 = OR + '/JKF_sar_surf_5.mat'
# SAR6 = OR + '/JKF_sar_surf_6.mat'
# SAR7 = OR + '/JKF_sar_surf_7.mat'
# SAR8 = OR + '/JKF_sar_surf_8.mat'
# SAR9 = OR + '/JKF_sar_surf_9.mat'
# SAR10 = OR +'/JKF_sar_surf_10.mat'
