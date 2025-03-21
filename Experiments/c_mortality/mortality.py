import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import io

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
from model_spiroclf import CNN
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler

# #ChanRev meth10A


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
parser.add('--sample_from_full_features', type=int, default=0, help = '(for use with bootstrap indices and args.save_option == 1) If 1, assumes that the precalculated features are the full features and samples from them according to args.val_idx_list. If 0, assumes that the precalculated features are sampled from bootstrap index.')
parser.add('--blow_filter', type=str, default='none', help = '"none" or "best", filter to use only best blow during training')
parser.add('--eob_method', type=str, default='max', help = 'zero, max, or none')
parser.add('--target', type=str, default='binary_0.7_threshold')
parser.add('--batch_size', type=int, default='512')
parser.add('--source_time_interval', type=int, default=10)
parser.add('--data_downsample_factor', type=int, default=5)
parser.add('--feature_volume_interval', type=int, default=50)
parser.add('-t','--transform', action='append', help='define transformations')
parser.add('--transform_p', type=float, default=0.5, help='probability of flow-volume or flow-time transform')
parser.add('--dataloader_workers', type=int, default=0, help='Parameter num_workers for dataloader')
parser.add('--max_length', type=int, default=300, help='max length of sample')
parser.add('--append_transform_flag', type=int, default=0, help='append transformation information to samples')

# Downstream Model
parser.add('--feat_option', type=int, default= 1, help='0: default, 1: feat avg, 2: best blow only')
parser.add('--qc_filter_code', type=int, default=12, help='for use when evaluating feature averaging.')
parser.add('--method', type=str, default='LogisticRegression')
parser.add('--pca', type=int, default=0)

# File Saving
parser.add('--model_path', type=str, default='simclr_faithful-glitter-1_bestloss.pt')
parser.add('--save_option', type=int, default=1, help='0: Save Features and Exit; 1: Load Features and run Linear Probe; 2: Run Everything (does not save features)')
parser.add('--save_downstream_model', type=int, default=0, help='0: False; 1: True')
args, unknown = parser.parse_known_args()

print_args(args)

if args.transform == None:
    transform_list = [0,1,2]
else:
    transform_list = [int(t) for t in args.transform]

if transform_list == [10]:
    transform_list = []

if args.feat_option == 0:
    # default, all blows
    blow_filter = 'qc'
    feat_avg = 0
    qc_filter_code = args.qc_filter_code 
    # should not be using blow sampling
    if 0 in transform_list:
        transform_list.remove(0)

elif args.feat_option == 1:
    # feature averaging
    blow_filter = 'qc'
    feat_avg = 1
    qc_filter_code = args.qc_filter_code  # note that filter_code = 12 does not filter anything and is the default.
elif args.feat_option == 2:
    # filter for best blow only
    blow_filter = 'best'
    feat_avg = 0
    qc_filter_code = None 

    # should not be using blow sampling
    if 0 in transform_list:
        transform_list.remove(0)


transform_p = args.transform_p
model_path = args.model_path
feat_option = args.feat_option
# run_name = 'dark-sun-465'

#####################################################################
# Load Model
#####################################################################
sys.path.append('./Quality/SimCLR')
device = torch.device("cuda:0" if torch.cuda.is_available()
                else "cpu")  # Use GPU if available
checkpoint = torch.load('./Files/models/spiroclf/' + args.model_path, map_location = device)
try:
    model_params = checkpoint['args']
    run_name = checkpoint['run_name']
    model = CNN( model_params.hidden_units, kernel_size = model_params.kernel_size, max_length = model_params.max_length, dropout_p=0, num_layers = model_params.num_layers, num_linear_units = model_params.num_linear_units, conv_masking = model_params.conv_masking, weight_norm = model_params.weight_norm, append_transform_flag=model_params.append_transform_flag)
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    model = torch.load('./Files/models/spiroclf/' + args.model_path, map_location = device) # older save file
model.to(device)
print('done!')

if (args.save_option == 0) or (args.save_option == 2):
    #####################################################################
    # Load Data
    #####################################################################
    files_path = './Files/'
    params = {
        'dataset': args.dataset,
        'max_length': args.max_length,
        'data_path': args.data_path,
        'id_path': args.id_path,
        'target': 'binary_0.7_threshold', 
        'blow_filter': blow_filter,
        'qc_filter_code': qc_filter_code,
        'transform': transform_list,
        'eob_method':'max',
        'idx_list': args.train_idx_list,
        'sample_size': None,
        'transform_p': transform_p,
        'downsample_factor': args.data_downsample_factor,
        'source_time_interval': args.source_time_interval,
        'feature_volume_interval': 50,
        'addn_metrics': True,
        'append_transform_flag': model_params.append_transform_flag,
    }
    dataset = load_dataset(**params)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.dataloader_workers)

    params['idx_list'] = args.val_idx_list
    t_dataset = load_dataset(**params)
    t_dataloader = DataLoader(t_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.dataloader_workers)
    aux_col = dataset.aux_col

    print('train samples: ' + str(len(dataset)))
    print('test samples: ' + str(len(t_dataset)))



    #####################################################################
    # Extract Features
    #####################################################################
    if feat_avg == 0:
        # no feature averaging
        feature_bank, feature_labels, aux_bank = extract_feats(model, dataloader)
        t_feature_bank, t_feature_labels, t_aux_bank = extract_feats(model, t_dataloader)
    if feat_avg == 1:
        # feature averaging
        feature_bank, feature_labels, aux_bank = extract_feats_avg(model, dataloader, n_aug_samples = 10, n_feat_input = model_params.max_length + 3 * model_params.append_transform_flag, n_feat_output = model_params.num_linear_units)
        t_feature_bank, t_feature_labels, t_aux_bank = extract_feats_avg(model, t_dataloader, n_aug_samples = 10, n_feat_input = model_params.max_length+ 3 * model_params.append_transform_flag, n_feat_output = model_params.num_linear_units)


    #####################################################################
    # Save Features
    #####################################################################

    save_path='./Files/Miscellaneous/SimCLR_Features/'
    prepend = '_'.join([
        run_name,
        args.dataset,
        str(args.feat_option),
        ''.join([str(t) for t in transform_list]),
        str(transform_p),
        args.val_idx_list,
        str(qc_filter_code),
    ])
    #z = prepend.split('_')
    #[int(char) for char in z[4]]

    print(save_path)
    torch.save(feature_bank, save_path + '%sfeature_bank.pt' % prepend)
    torch.save(feature_labels, save_path + '%sfeature_labels.pt' % prepend)
    torch.save(aux_bank, save_path + '%saux_bank.pt' % prepend)
    torch.save(t_feature_bank, save_path + '%st_feature_bank.pt' % prepend)
    torch.save(t_feature_labels, save_path + '%st_feature_labels.pt' % prepend)
    torch.save(t_aux_bank, save_path + '%st_aux_bank.pt' % prepend)
    save_dict(aux_col, save_path+'aux_col_%s.pkl' % args.dataset)
    if args.save_option == 0: sys.exit()

if args.save_option == 1:

    #load bootstrap dictionary if args.sample_from_full_features == 1
    if args.sample_from_full_features == 1:
        parse = args.val_idx_list.split('_')
        idx = int(parse[2])
        split = parse[1]

        id_path = args.id_path
        bootstrap_dict = load_dict(os.path.join(id_path, 'bootstrap_dict_v2.pkl'))
        filter_ids = bootstrap_dict[idx][split]
        filter_ids = pd.DataFrame(filter_ids, columns = ['ID'])
    else:
        split = args.val_idx_list

    # load features
    save_path='./Files/Miscellaneous/SimCLR_Features/'
    prepend = '_'.join([
        run_name,
        args.dataset,
        str(args.feat_option),
        ''.join([str(t) for t in transform_list]),
        str(transform_p),
        split,
        str(qc_filter_code),
    ])

    device = 'cpu'
    feature_bank = torch.load(save_path+'%sfeature_bank.pt' % prepend, map_location = device)
    feature_labels = torch.load(save_path+'%sfeature_labels.pt' % prepend, map_location = device)
    aux_bank = torch.load(save_path+'%saux_bank.pt' % prepend, map_location = device)
    t_feature_bank = torch.load(save_path+'%st_feature_bank.pt' % prepend, map_location = device)
    t_feature_labels = torch.load(save_path+'%st_feature_labels.pt' % prepend, map_location = device)
    t_aux_bank = torch.load(save_path+'%st_aux_bank.pt' % prepend, map_location = device)

    aux_col = load_dict(save_path+'aux_col_%s.pkl' % args.dataset)

    #filter bootstrap values if args.sample_from_full_features == 1
    if args.sample_from_full_features == 1:
        t_id_list = pd.DataFrame(t_aux_bank[:,aux_col.index('ID')].long(), columns = ['ID']).reset_index() # dataframe that maps IDs in index values for t_feature_bank
        filter_idx = t_id_list.merge(filter_ids, on = 'ID', how = 'right') # np array of indices to filter t_feature_bank
        
        # identify missing IDs. These sometimes occur when filtering based on QC code, since not all participants have PFTs with the QC code.
        missing_id = filter_idx[filter_idx['index'].isna()]['ID'].unique()
        if missing_id.shape[0] > 0:
            for j in range(missing_id.shape[0]):
                print('missing IDs: ' + str(missing_id[j]))

        filter_idx = filter_idx['index']
        filter_idx = filter_idx[~filter_idx.isna()].values

        # filter features
        t_feature_bank = t_feature_bank[filter_idx,:]
        t_feature_labels = t_feature_labels[filter_idx]
        t_aux_bank = t_aux_bank[filter_idx, :]


# get ethnicities for test set
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
# Preprocess Features
#####################################################################
# each row of the dataset is a single blow (2-3 blows per individual)

if feat_avg == 1:
    use_aux = 0
else:
    use_aux = 0

# append rejection information
if use_aux == 1:
    aux_onehot = aux_bank[:,aux_col.index('qc4')]  # qc4 = 1 indicates the blow passed QC
    aux_onehot = aux_onehot.reshape(feature_bank.shape[0],-1)
    x = tensor2numpy(torch.cat((aux_onehot, feature_bank), dim = 1))
    

    aux_onehot = t_aux_bank[:,aux_col.index('qc4')]  # qc4 = 1 indicates the blow passed QC
    aux_onehot = aux_onehot.reshape(t_feature_bank.shape[0],-1)
    t_x = tensor2numpy(torch.cat((aux_onehot, t_feature_bank), dim = 1))
else:
    x = tensor2numpy(feature_bank)
    t_x = tensor2numpy(t_feature_bank)


# Preprocessing
# Scale Spiro Features
sc = StandardScaler()
sc.fit(x)
x = sc.transform(x)
t_x = sc.transform(t_x)

# PCA on Spiro Features
if args.pca == 1:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 0.99)
    # pca = PCA(n_components = 16)
    pca.fit(x)
    x = pca.transform(x)
    t_x = pca.transform(t_x)

# survival features
y = aux_bank[:,aux_col.index('event')]
time_to_event = aux_bank[:,aux_col.index('time_to_event')]
t_y = t_aux_bank[:,aux_col.index('event')]
t_time_to_event = t_aux_bank[:,aux_col.index('time_to_event')]



#####################################################################
# Survival Model
#####################################################################
from sksurv.linear_model import CoxPHSurvivalAnalysis
sklearn_labels = np.concatenate((tensor2numpy(y).reshape(-1,1), tensor2numpy(time_to_event).reshape(-1,1)), axis = 1)
sklearn_labels = np.core.records.fromarrays(sklearn_labels.transpose(), names='event, time_to_event',formats = '?, i8')
estimator = CoxPHSurvivalAnalysis().fit(x,sklearn_labels)


#####################################################################
# Evaluate Performance
#####################################################################

path = './Files/Results/SimCLR_Survival'
t_sklearn_labels = np.concatenate((tensor2numpy(t_y).reshape(-1,1), tensor2numpy(t_time_to_event).reshape(-1,1)), axis = 1)
t_sklearn_labels = np.core.records.fromarrays(t_sklearn_labels.transpose(), names='event, time_to_event', formats = '?, i8')

# iterative over different test subsets
subset_list = [torch.ones_like(eth).bool(), (eth).bool(), (1-eth).bool()]
save_path_list = ['', 'eth-cau', 'eth-oth']

for idx_list, save_path in zip(subset_list, save_path_list):

    new_path = os.path.join(path,save_path) # save results in separate folders
    make_dir(new_path)

    # concordance score
    c_score = estimator.score(t_x[idx_list, :], t_sklearn_labels[idx_list])
    print('concordance score: '+ str(c_score))


    # integrated brier score
    from sksurv.metrics import integrated_brier_score
    survs = estimator.predict_survival_function(t_x)
    times = np.arange(1, 10)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    brier_score = integrated_brier_score(sklearn_labels, t_sklearn_labels[idx_list], preds[idx_list, :], times)
    print('integrated brier score: ' + str(brier_score))

    c_score_dict = {}
    brier_dict = {}
    c_score_dict[args.val_idx_list] = c_score
    brier_dict[args.val_idx_list] = brier_score

    prepend = '_'.join([
        run_name,
        args.dataset,
        str(args.feat_option),
        ''.join([str(t) for t in transform_list]),
        str(transform_p),
        args.val_idx_list,
        str(qc_filter_code),
        str(args.sample_from_full_features),
        'pca'+str(args.pca),
    ])
    save_dict(c_score_dict, os.path.join(new_path, 'survival_cscore_%s' % prepend))
    save_dict(brier_dict, os.path.join(new_path, 'survival_brier_%s' % prepend))

print('done!')