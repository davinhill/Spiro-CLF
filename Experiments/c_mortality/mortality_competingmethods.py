
import os
from tkinter import W
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# This file saves the aux_bank file for comparison calculations

import io
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_theme()
sys.path.append('./')
from utils import *
from load_data import load_dataset

import configargparse
parser = configargparse.ArgParser(config_file_parser_class = configargparse.YAMLConfigFileParser)
parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

# Dataset
parser.add('--dataset', type=str, default='ukbb', help = 'ukbb or copdgene')
parser.add('--files_path', type=str, default='/udd/redhi/spiro_dev/Files', help = 'path to files folder')
parser.add('--data_path', type=str, default='/udd/redhi/spiro_dev/Files/3blows_with_max_rejection.csv', help = 'path to files folder')
parser.add('--id_path', type=str, default='./Files/quality_train_test_ids/ukbb', help = 'path to files folder')
parser.add('--train_idx_list', type=str, default='train_50k', help = 'which set of IDs to use during training. "train", "train_val", "train_50k"')
parser.add('--val_idx_list', type=str, default='val_35k', help = 'which set of IDs to use during testing. "val", "test", "val_35k"')
parser.add('--eob_method', type=str, default='max', help = 'zero, max, or none')
parser.add('--target', type=str, default='binary_0.7_threshold')
parser.add('--batch_size', type=int, default='512')
parser.add('--source_time_interval', type=int, default=10)
parser.add('--data_downsample_factor', type=int, default=5)
parser.add('--feature_volume_interval', type=int, default=50)
parser.add('-t','--transform', action='append', help='define transformations')
parser.add('--transform_p', type=float, default=0.5, help='probability of flow-volume or flow-time transform')
parser.add('--dataloader_workers', type=int, default=0, help='Parameter num_workers for dataloader')

parser.add('--blow_filter', type=str, default='best', help = '"none" or "best", filter to use only best blow during training')
parser.add('--qc_filter_code', type=int, default=12, help='for use when evaluating feature averaging.')

args, unknown = parser.parse_known_args()
print_args(args)


if args.transform == None:
    transform_list = [0,1]
else:
    transform_list = [int(t) for t in args.transform]

# filter for best blow only
blow_filter = args.blow_filter
qc_filter_code = args.qc_filter_code

# should not be using blow sampling
if 0 in transform_list: transform_list.remove(0)

#### Append Identifying Information
#####################################################################
# Load Data
#####################################################################
files_path = './Files/'
params = {
    'dataset': args.dataset,
    'max_length': 300,
    'data_path': args.data_path,
    'id_path': args.id_path,
    'target': 'binary_0.7_threshold', 
    'blow_filter': blow_filter,
    'qc_filter_code': qc_filter_code,
    'transform': transform_list,
    'eob_method':'max',
    'idx_list': args.train_idx_list,
    'sample_size': None,
    'transform_p': 0.5,
    'downsample_factor': args.data_downsample_factor,
    'source_time_interval': args.source_time_interval,
    'feature_volume_interval': 50,
    'addn_metrics': True,
}
dataset = load_dataset(**params)
dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers=args.dataloader_workers)

params['idx_list'] = args.val_idx_list
t_dataset = load_dataset(**params)
t_dataloader = DataLoader(t_dataset, batch_size = args.batch_size, shuffle = False, num_workers=args.dataloader_workers)
aux_col = dataset.aux_col

print('train samples: ' + str(len(dataset)))
print('test samples: ' + str(len(t_dataset)))


#####################################################################
# Extract Features
#####################################################################
# no feature averaging
print('Extracting Features...')
model = None
feature_bank, feature_labels, aux_bank = extract_feats(model, dataloader, aux_only = True)
t_feature_bank, t_feature_labels, t_aux_bank = extract_feats(model, t_dataloader, aux_only = True)

# # save aux bank for competitor comparison
# path = './Files/Miscellaneous/SimCLR_Features/'
# torch.save(aux_bank, path + 'aux_bank_comparison_%s_%s.pt' % (args.dataset, args.val_idx_list))
# torch.save(t_aux_bank, path + 't_aux_bank_comparison_%s_%s_t.pt' % (args.dataset, args.val_idx_list))

print('done!')

#####################################################################
# Get ethnicities for test set
#####################################################################

if args.dataset == 'ukbb':
    eth = pd.DataFrame(t_aux_bank[:,aux_col.index('ethnicity_selfreported')])
    eth[0] = eth[0].astype('str')
    eth[0] = (eth[0].str[:1]=='1')*1 # 1: caucasian, 0: other
    eth = numpy2cuda(eth[0].values)

else:
    copdgene_id_mapping = pd.read_csv('./Files/quality_train_test_ids/copdgene/id_sid_mapping.csv', index_col = 0)
    cggrid = pd.read_csv('./Files/cgGridPheno.csv', sep = ',') # covariate values
    covar_matrix = copdgene_lookup_covar(t_aux_bank, aux_col, cggrid, copdgene_id_mapping)
    eth = (covar_matrix.loc[:,'race'].values == 1)*1
    eth = numpy2cuda(eth)

#####################################################################
# Calculate Summary Metrics
#####################################################################

# get fvc, fev1 information
fvc = aux_bank[:,aux_col.index('fvc')]
fev1 = aux_bank[:,aux_col.index('fev1')]
ratio = fev1/fvc
fev1_fvc_ratio = torch.cat((fvc.reshape(-1,1), fev1.reshape(-1,1), ratio.reshape(-1,1)), dim = 1)
fev2 = aux_bank[:,aux_col.index('fev2')]
fev3 = aux_bank[:,aux_col.index('fev3')]
fev6 = aux_bank[:,aux_col.index('fev6')]
fev1_fev3  = replace_nan(fev1/fev3, 0)
fev1_fev6 = replace_nan(fev1 / fev6, 0)
fev1_fev6_fev6 = torch.cat((fev1_fev6.reshape(-1,1), fev6.reshape(-1,1)), dim = 1)
fef25 = aux_bank[:,aux_col.index('fef25')]
fef75 = aux_bank[:,aux_col.index('fef75')]
fef2575 = fef25 - fef75

t_fvc = t_aux_bank[:,aux_col.index('fvc')]
t_fev1 = t_aux_bank[:,aux_col.index('fev1')]
t_ratio = t_fev1 / t_fvc
t_fev1_fvc_ratio = torch.cat((fvc.reshape(-1,1), fev1.reshape(-1,1), ratio.reshape(-1,1)), dim = 1)
t_fev2 = t_aux_bank[:,aux_col.index('fev2')]
t_fev3 = t_aux_bank[:,aux_col.index('fev3')]
t_fev6 = t_aux_bank[:,aux_col.index('fev6')]
t_fev1_fev3  = replace_nan(t_fev1/t_fev3, 0)
t_fev1_fev6 = replace_nan(t_fev1 / t_fev6, 0)
t_fev1_fev6_fev6 = torch.cat((t_fev1_fev6.reshape(-1,1), t_fev6.reshape(-1,1)), dim = 1)
t_fef25 = t_aux_bank[:,aux_col.index('fef25')]
t_fef75 = t_aux_bank[:,aux_col.index('fef75')]
t_fef2575 = t_fef25 - t_fef75


####################################3
y = aux_bank[:,aux_col.index('event')]
time_to_event = aux_bank[:,aux_col.index('time_to_event')]
t_y = t_aux_bank[:,aux_col.index('event')]
t_time_to_event = t_aux_bank[:,aux_col.index('time_to_event')]

####################################3
test_list = []
t_test_list = []
name_list = []

# fvc
x = tensor2numpy(fvc.reshape(-1,1))
t_x = tensor2numpy(t_fvc.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FVC')

# fev1
x = tensor2numpy(fev1.reshape(-1,1))
t_x = tensor2numpy(t_fev1.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1')

# ratio
x = tensor2numpy(ratio.reshape(-1,1))
t_x = tensor2numpy(t_ratio.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1/FVC')


# all three
x = torch.cat((fvc.reshape(-1,1), fev1.reshape(-1,1), ratio.reshape(-1,1)), dim = 1)
x = tensor2numpy(x)
t_x = torch.cat((t_fvc.reshape(-1,1), t_fev1.reshape(-1,1), t_ratio.reshape(-1,1)), dim = 1)
t_x = tensor2numpy(t_x)
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1, FVC, FEV1/FVC')

# fev2
x = tensor2numpy(fev2.reshape(-1,1))
t_x = tensor2numpy(t_fev2.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV2')


# fev3
x = tensor2numpy(fev3.reshape(-1,1))
t_x = tensor2numpy(t_fev3.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV3')


# fev6
x = tensor2numpy(fev6.reshape(-1,1))
t_x = tensor2numpy(t_fev6.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV6')


# fev1_fev3
x = tensor2numpy(fev1_fev3.reshape(-1,1))
t_x = tensor2numpy(t_fev1_fev3.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1/FEV3')


# fev1_fev6
x = tensor2numpy(fev1_fev6.reshape(-1,1))
t_x = tensor2numpy(t_fev1_fev6.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1/FEV6')


# fev1_fev6_fev6
x = tensor2numpy(fev1_fev6_fev6)
t_x = tensor2numpy(t_fev1_fev6_fev6)
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1/FEV6, FEV6')


# fef25
x = tensor2numpy(fef25.reshape(-1,1))
t_x = tensor2numpy(t_fef25.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEF25')


# fef75
x = tensor2numpy(fef75.reshape(-1,1))
t_x = tensor2numpy(t_fef75.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEF75')

# fef2575
x = tensor2numpy(fef2575.reshape(-1,1))
t_x = tensor2numpy(t_fef2575.reshape(-1,1))
test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEF25-FEF75')

# all
x = torch.cat((fvc.reshape(-1,1), fev1.reshape(-1,1), ratio.reshape(-1,1), fev1_fev6_fev6, fef2575.reshape(-1,1)), dim = 1)
x = tensor2numpy(x)
t_x = torch.cat((t_fvc.reshape(-1,1), t_fev1.reshape(-1,1), t_ratio.reshape(-1,1), t_fev1_fev6_fev6, t_fef2575.reshape(-1,1)), dim = 1)
t_x = tensor2numpy(t_x)

test_list.append(x)
t_test_list.append(t_x)
name_list.append('FEV1, FVC, FEV1/FVC, FEV1/FEV6, FEV6, FEF25-75')



#####################################################################
# Survival Model
#####################################################################

metric_list = [
    'FEV1',
    'FVC',
    'FEV1/FVC',
    'FEV1, FVC, FEV1/FVC',
    'FEV6',
    'FEV1/FEV6, FEV6',
    'FEF25-FEF75',
    'FEV1, FVC, FEV1/FVC, FEV1/FEV6, FEV6, FEF25-75',
]


for metric_idx,test_metric in enumerate(metric_list):
    idx = name_list.index(test_metric)
    x = test_list[idx]
    t_x = t_test_list[idx]

    print('=======================')
    print(test_metric)

    from sksurv.linear_model import CoxPHSurvivalAnalysis
    sklearn_labels = np.concatenate((tensor2numpy(y).reshape(-1,1), tensor2numpy(time_to_event).reshape(-1,1)), axis = 1)
    sklearn_labels = np.core.records.fromarrays(sklearn_labels.transpose(), names='event, time_to_event',formats = '?, i8')
    try:
        estimator = CoxPHSurvivalAnalysis().fit(x,sklearn_labels)
    except:
        import pdb; pdb.set_trace()

    # concordance score
    t_sklearn_labels = np.concatenate((tensor2numpy(t_y).reshape(-1,1), tensor2numpy(t_time_to_event).reshape(-1,1)), axis = 1)
    t_sklearn_labels = np.core.records.fromarrays(t_sklearn_labels.transpose(), names='event, time_to_event', formats = '?, i8')

    ########################################
    # iterative over different test subsets
    subset_list = [torch.ones_like(eth).bool(), (eth).bool(), (1-eth).bool()]
    save_path_list = ['', 'eth-cau', 'eth-oth']

    for idx_list, save_path in zip(subset_list, save_path_list):

        path = './Files/Results/SimCLR_Survival/%s/comparison' % save_path
        make_dir(path)

        score = estimator.score(t_x[idx_list, :], t_sklearn_labels[idx_list])
        print('concordance score: '+ str(score))

        '''
        # brier score
        from sksurv.metrics import brier_score
        survs = estimator.predict_survival_function(t_x)

        times = np.arange(1, 10)
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
        times, score = brier_score(sklearn_labels, t_sklearn_labels, preds, times)
        print('brier score: ' + str(score))
        '''

        c_score_dict = {}
        c_score_dict[args.val_idx_list] = score
        if os.path.basename(os.path.normpath(args.id_path)) == 'ukbb_sitesplit':
            sitesplit_tag = 'sitesplit/'
        else:
            sitesplit_tag = ''

        filename = os.path.join(path, '%ssurvival_c_score_metric%s_%s_%s_%s' % (sitesplit_tag, str(metric_idx), args.dataset, args.val_idx_list, str(qc_filter_code)))
        save_dict(c_score_dict, filename)


        print(filename)
        print('done!')