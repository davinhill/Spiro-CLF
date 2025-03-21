# #ChanRev meth10B

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

import configargparse
parser = configargparse.ArgParser(config_file_parser_class = configargparse.YAMLConfigFileParser)
parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

# Dataset
parser.add('--dataset', type=str, default='ukbb', help = 'ukbb or copdgene')
parser.add('--files_path', type=str, default='./Files', help = 'path to files folder')
parser.add('--data_path', type=str, default='./Files/COPDGene/quality_copdgene_dataset_v2.csv', help = 'path to files folder')
parser.add('--id_path', type=str, default='./Files/quality_train_test_ids/ukbb', help = 'path to files folder')
parser.add('--train_idx_list', type=str, default='train', help = 'which set of IDs to use during training. "train", "train_val", "train_50k"')
parser.add('--val_idx_list', type=str, default='bootstrap_test_0', help = 'which set of IDs to use during testing. "val", "test", "val_35k"')
parser.add('--sample_from_full_features', type=int, default=1, help = '(for use with bootstrap indices and args.save_option == 1) If 1, assumes that the precalculated features are the full features and samples from them according to args.val_idx_list. If 0, assumes that the precalculated features are sampled from bootstrap index.')
parser.add('--blow_filter', type=str, default='none', help = '"none" or "best", filter to use only best blow during training')
parser.add('--eob_method', type=str, default='max', help = 'zero, max, or none')
parser.add('--target', type=str, default='binary_0.7_threshold')
parser.add('--batch_size', type=int, default='512')
parser.add('--source_time_interval', type=int, default=10)
parser.add('--data_downsample_factor', type=int, default=5)
parser.add('--feature_volume_interval', type=int, default=50)
parser.add('-t','--transform', action='append', help='define transformations')
parser.add('--transform_p', type=float, default=0.5, help='probability of flow-volume or flow-time transform')
parser.add('--dataloader_workers', type=int, default=8, help='Parameter num_workers for dataloader')
parser.add('--max_length', type=int, default=300, help='max length of sample')
parser.add('--append_transform_flag', type=int, default=0, help='append transformation information to samples')

# Downstream Model
parser.add('--feat_option', type=int, default= 0, help='0: default, 1: feat avg, 2: best blow only')
parser.add('--qc_filter_code', type=int, default=12, help='for use when evaluating feature averaging.')
parser.add('--method', type=str, default='LogisticRegression')

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

if args.feat_option == 0:
    # default, no filter
    blow_filter = args.blow_filter
    feat_avg = 0
    qc_filter_code = args.qc_filter_code  # note that filter_code = 12 does not filter anything and is the default.
    # should not be using blow sampling
    if 0 in transform_list:
        transform_list.remove(0)

elif args.feat_option == 1:
    # feature averaging
    blow_filter = args.blow_filter
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
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)

    params['idx_list'] = args.val_idx_list
    t_dataset = load_dataset(**params)
    t_dataloader = DataLoader(t_dataset, batch_size = args.batch_size, shuffle = False)
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

        prepend = '_'.join([
            run_name,
            args.dataset,
            str(args.feat_option),
            ''.join([str(t) for t in transform_list]),
            str(transform_p),
            split,
            str(12), # loads all blows
        ])

    else:
        prepend = '_'.join([
            run_name,
            args.dataset,
            str(args.feat_option),
            ''.join([str(t) for t in transform_list]),
            str(transform_p),
            args.val_idx_list,
            str(qc_filter_code),
        ])

    # load features
    save_path='./Files/Miscellaneous/SimCLR_Features/'
    device = 'cpu'
    feature_bank = torch.load(save_path+'%sfeature_bank.pt' % prepend, map_location = device)
    feature_labels = torch.load(save_path+'%sfeature_labels.pt' % prepend, map_location = device)
    aux_bank = torch.load(save_path+'%saux_bank.pt' % prepend, map_location = device)
    t_feature_bank = torch.load(save_path+'%st_feature_bank.pt' % prepend, map_location = device)
    t_feature_labels = torch.load(save_path+'%st_feature_labels.pt' % prepend, map_location = device)
    t_aux_bank = torch.load(save_path+'%st_aux_bank.pt' % prepend, map_location = device)

    aux_col = load_dict(save_path+'aux_col_%s.pkl' % args.dataset)

    if args.sample_from_full_features == 1:
        # 1. Filter based on QC code
        qc_idx = filter_qc(dataset = args.dataset, data_aux = t_aux_bank, aux_col = aux_col, qc_filter_code = args.qc_filter_code)

        # filter features
        t_feature_bank = t_feature_bank[qc_idx,:]
        t_feature_labels = t_feature_labels[qc_idx]
        t_aux_bank = t_aux_bank[qc_idx, :]


        # 2. Bootstrap resampling based on selected bootstrap IDs. Resampling is performed on participant IDs; all blows from the selected ID are included in the resampled dataset.
        
        # iterate through IDs in the bootstrap ID list while concatenating the PFTs associated with each ID
        filter_idx = []
        t_id_list = pd.DataFrame(t_aux_bank[:,aux_col.index('ID')].long(), columns = ['ID']).reset_index()
        for k in range(filter_ids.shape[0]):
            id = filter_ids.iloc[k,0] # select ID
            filter_idx.append(t_id_list.loc[t_id_list['ID'] == id, 'index'].values) # append PFTs associated with the ID
        filter_idx = np.hstack(filter_idx) # convert to np array

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

#################################################
# calculate targets (fev1 percent predicted)
#################################################

if args.dataset == 'ukbb':
    fev1_pp = ukbb_fev1pp(aux_bank, aux_col)
    t_fev1_pp = ukbb_fev1pp(t_aux_bank, aux_col)

elif args.dataset == 'copdgene':
    copdgene_id_mapping = pd.read_csv('./Files/quality_train_test_ids/copdgene/id_sid_mapping.csv', index_col = 0)
    cggrid = pd.read_csv('./Files/cgGridPheno.csv', sep = ',') # covariate values
    fev1_pp = copdgene_fev1pp(aux_bank, aux_col, cggrid, copdgene_id_mapping)
    t_fev1_pp = copdgene_fev1pp(t_aux_bank, aux_col, cggrid, copdgene_id_mapping)



# save information for identifying results
tmp = prepend.split('_')
model_name = tmp[1]
val_option = tmp[2][:-3]

# Functions for saving results

#### Append Identifying Information
def append_experiment_info(df, model_path, feat_option, method, use_aux, metric, qc_split = None):
    df['experiment'] = 'predict_binary_ratio'
    df['model_path'] = model_path
    df['feat_option'] = feat_option
    df['use_aux'] = use_aux
    df['method'] = method
    df['metric'] = metric
    df['test_set'] = args.val_idx_list
    df['qc_filter_code'] = qc_filter_code
    df['date_updated'] = str(datetime.now())
    if qc_split is not None:
        df = pd.concat([df, qc_split], axis = 1)
    return df


def evaluate_LP(pred, scores, t_feature_labels, t_aux_bank, aux_col):
    correct = pred.eq(tensor2cuda(t_feature_labels).view_as(tensor2cuda(pred))).sum().item()
    accy = (correct/t_feature_labels.shape[0])
    auc = roc_auc_score(y_true = tensor2numpy(t_feature_labels), y_score = scores)
    f1 = f1_score(tensor2numpy(t_feature_labels), tensor2numpy(pred))
    precision = precision_score(tensor2numpy(t_feature_labels), tensor2numpy(pred))
    recall = recall_score(tensor2numpy(t_feature_labels), tensor2numpy(pred))
    fpr, tpr, _ = roc_curve(tensor2numpy(t_feature_labels),  scores)

    aux_auc = np.array([auc]).reshape(1,-1)
    aux_accy = np.array([accy]).reshape(1,-1)
    n_ID = np.unique(t_aux_bank[:,aux_col.index('ID')]).shape[0]
    #aux_counter = np.array([t_feature_labels.shape[0]]).reshape(1,-1)
    aux_counter = np.array([n_ID]).reshape(1,-1)
    aux_f1 = np.array([f1]).reshape(1,-1)
    aux_precision = np.array([precision]).reshape(1,-1)
    aux_recall = np.array([recall]).reshape(1,-1)

    output_aux_accy = pd.DataFrame((aux_accy.reshape(1,-1)), columns = ['score'])
    output_aux_auc = pd.DataFrame(aux_auc, columns = ['score'])
    output_counter = pd.DataFrame(aux_counter, columns = ['score'])
    output_aux_f1 = pd.DataFrame(aux_f1, columns = ['score'])
    output_aux_precision = pd.DataFrame(aux_precision, columns = ['score'])
    output_aux_recall = pd.DataFrame(aux_recall, columns = ['score'])

    return output_aux_accy, output_aux_auc, output_counter, output_aux_f1, output_aux_precision, output_aux_recall, [fpr, tpr]


def evaluate_LP_aux(pred, scores, t_feature_labels, t_aux_bank, aux_col):
    aux_auc = []
    aux_accy = []
    aux_counter = []
    
    # additional filters
    # qc0.1 User Rejected
    tmp = (1- t_aux_bank[:,aux_col.index('USER_REJECTED')]).reshape(-1,1).int().numpy()  # 1 if not rejected
    aux_addn = pd.DataFrame(tmp, columns = ['qc0.1']).astype('int16')

    # qc0.2 Start of test
    tmp =  (1- t_aux_bank[:,aux_col.index('START_OF_TEST')]).reshape(-1,1).int().numpy() # 1 if not rejected
    aux_addn['qc0.2'] = tmp
    aux_addn['qc0.2'] = aux_addn['qc0.2'] * aux_addn['qc0.1']

    # qc0.3 Time to PEF
    tmp =  (1- t_aux_bank[:,aux_col.index('TIME_TO_PEF')]).reshape(-1,1).int().numpy() # 1 if not rejected
    aux_addn['qc0.3'] = tmp
    aux_addn['qc0.3'] = aux_addn['qc0.3'] * aux_addn['qc0.2']

    # qc0.4 Coughing
    tmp =  (1- t_aux_bank[:,aux_col.index('COUGHING')]).reshape(-1,1).int().numpy() # 1 if not rejected
    aux_addn['qc0.4'] = tmp
    aux_addn['qc0.4'] = aux_addn['qc0.4'] * aux_addn['qc0.3']

    '''
    # Note: This is the same as QC1
    # qc0.5 End of Test
    tmp =  (1- t_aux_bank[:,aux_col.index('END_OF_TEST')]).reshape(-1,1).int().numpy() # 1 if not rejected
    aux_addn['qc0.5'] = tmp
    aux_addn['qc0.5'] = aux_addn['qc0.5'] * aux_addn['qc0.4']
    '''

    # append to aux_bank
    t_aux_bank = torch.cat((t_aux_bank, torch.tensor(aux_addn.values)), axis = 1)
    aux_col = aux_col + [
        'qc0.1',
        'qc0.2',
        'qc0.3',
        'qc0.4',
        #'qc0.5',
    ]

    roc_dict = {}
    rejection_labels = [
        'QC0.1_reject',
        'QC0.2_reject',
        'QC0.3_reject',
        'QC0.4_reject',
        'QC1_reject',
        'QC2_reject',
        'QC3_reject',
        'QC4_reject',
        'non_maximal',
        'best_blow',
        'all_ex_best',
        'all_nonrejected',
        'total',
    ]
    for aux in range(13):

        # filter
        import pdb; pdb.set_trace()
        if aux == 0:
            aux_idx = torch.where(t_aux_bank[:,aux_col.index('qc0.1')] == 0)[0]
        elif aux == 1:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc0.2')] == 0) * (t_aux_bank[:,aux_col.index('qc0.1')] == 1))[0]
        elif aux == 2:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc0.3')] == 0) * (t_aux_bank[:,aux_col.index('qc0.2')] == 1))[0]
        elif aux == 3:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc0.4')] == 0) * (t_aux_bank[:,aux_col.index('qc0.3')] == 1))[0]
        elif aux == 4:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc1')] == 0) * (t_aux_bank[:,aux_col.index('qc0.4')] == 1))[0]
        elif aux == 5:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc2')] == 0) * (t_aux_bank[:,aux_col.index('qc1')] == 1))[0]
        elif aux == 6:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc3')] == 0) * (t_aux_bank[:,aux_col.index('qc2')] == 1))[0]
        elif aux == 7:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('qc4')] == 0) * (t_aux_bank[:,aux_col.index('qc3')] == 1))[0]
        elif aux == 8:
            aux_idx = torch.where((t_aux_bank[:,aux_col.index('max_flag')] == 0) * (t_aux_bank[:,aux_col.index('qc4')] == 1))[0] # passed qc but not best blow (non-maximal)
        elif aux == 9:
            aux_idx = torch.where(t_aux_bank[:,aux_col.index('max_flag')] == 1)[0] # best blow
        elif aux == 10:
            aux_idx = torch.where(t_aux_bank[:,aux_col.index('max_flag')] == 0)[0] # all excluding best blow
        elif aux == 11:
            aux_idx = torch.where(t_aux_bank[:,aux_col.index('qc4')] == 1)[0] # all nonrejected
        elif aux == 12:
            aux_idx = torch.where(t_aux_bank[:,aux_col.index('ID')] != 0)[0] # all blows
        aux_counter.append(aux_idx.shape[0])
        ###############
        # Accuracy
        pred_tmp = pred[tensor2numpy(aux_idx)]
        y_tmp = t_feature_labels[aux_idx]
        if y_tmp.shape[0] == 0:
            accy = -1
        else:
            try:
                correct = pred_tmp.eq(y_tmp.view_as(pred_tmp)).sum().item()
                accy = (100.*correct/y_tmp.shape[0])
            except:
                accy = -1
                warnings.warn('skipped accy calculation')
        aux_accy.append(accy)
        ###############
        # AUC
        if len(aux_idx) == 0:
            aux_auc.append(-1)
        else:
            try:
                scores_tmp = scores[tensor2numpy(aux_idx)] # filter predicted probabilities for samples in the auxillary class
                y_tmp = t_feature_labels[aux_idx] # filter predicted probabilities for samples in the auxillary class
                aux_auc.append(roc_auc_score(y_true = tensor2numpy(y_tmp), y_score = scores_tmp))
                fpr, tpr, _ = roc_curve(tensor2numpy(y_tmp),  scores_tmp)
                roc_dict[rejection_labels[aux]] = [fpr, tpr]
            except:
                aux_auc.append(-1)
                warnings.warn('skipped auc calculation (likely due to only one class)')
            finally:
                pass
    aux_auc = np.array(aux_auc).reshape(1,-1)
    aux_accy = np.array(aux_accy).reshape(1,-1)
    aux_counter = np.array(aux_counter).reshape(1,-1)

    ###########
    # formatting
    output_aux_accy = pd.DataFrame(aux_accy, columns = rejection_labels)
    output_aux_auc = pd.DataFrame(aux_auc, columns = rejection_labels)
    output_aux_counter = pd.DataFrame(aux_counter, columns = rejection_labels)

    return output_aux_accy, output_aux_auc, output_aux_counter, roc_dict

#####################################################################
# Preprocess Features
#####################################################################
# each row of the dataset is a single blow (2-3 blows per individual)

'''
if feat_option == 1:
    use_aux = 0
else:
    use_aux = 1
'''
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

feature_labels = numpy2cuda((fev1_pp < 80)*1)
t_feature_labels = numpy2cuda((t_fev1_pp < 80)*1)


method = args.method
######################################################
if method == 'RidgeRegression':
    ##### Train Model
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler
    #model = linear_model.LogisticRegression(solver='liblinear')

    # Scale Data
    sc = StandardScaler()
    sc.fit(x)
    x_scale = sc.transform(x)
    t_x_scale = sc.transform(t_x)

    model = linear_model.LogisticRegressionCV(penalty = 'l2', solver = 'liblinear', random_state=0, n_jobs=-1, max_iter = 1e4)
    model.fit(x_scale, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x_scale)
    pred = torch.from_numpy(pred)
    scores = model.predict_proba(t_x_scale)[:,1]


######################################################
elif method == 'LogisticRegression':
    # train linear model
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler

    # Scale Data
    sc = StandardScaler()
    sc.fit(x)
    x_scale = sc.transform(x)
    t_x_scale = sc.transform(t_x)

    model = linear_model.LogisticRegression(penalty = 'none', solver = 'saga', random_state = 0, n_jobs = -1,max_iter = int(1e4))
    model.fit(x_scale, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x_scale)
    pred = numpy2cuda(pred)
    scores = model.predict_proba(t_x_scale)[:,1]
    

######################################################
elif method == 'LASSO':

    # train linear model
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler

    # Scale Data
    sc = StandardScaler()
    sc.fit(x)
    x_scale = sc.transform(x)
    t_x_scale = sc.transform(t_x)

    model = linear_model.LogisticRegressionCV(penalty = 'l1', solver = 'liblinear', random_state=0, n_jobs=-1,max_iter = 1e6, tol=0.001)
    model.fit(x_scale, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x_scale)
    pred = torch.from_numpy(pred)
    scores = model.predict_proba(t_x_scale)[:,1]


######################################################
elif method == 'RandomForest':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(max_depth=10, random_state=0)
    model.fit(x, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x)
    pred = numpy2cuda(pred)
    scores = model.predict_proba(t_x)[:,1]


######################################################
elif method == 'NaiveBayes':

    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(x, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x)
    pred = numpy2cuda(pred)
    scores = model.predict_proba(t_x)[:,1]

######################################################
elif method == 'LDA':

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    model = LDA()
    model.fit(x, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x)
    pred = numpy2cuda(pred)
    scores = model.predict_proba(t_x)[:,1]


######################################################
elif method == 'QDA':

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

    model = QDA()
    model.fit(x, tensor2numpy(feature_labels.int()))
    pred = model.predict(t_x)
    pred = numpy2cuda(pred)
    scores = model.predict_proba(t_x)[:,1]

path = './Files/Results/SimCLR_Probe_FEV1pp_binary_bootstrap_v3'
downstream_model_path = './Files/models'

if args.save_downstream_model == 1:
    save_dict(model, path = downstream_model_path + '/explain_fev1ppbinary_%s.pkl' % (method))


# iterative over different test subsets
subset_list = [torch.ones_like(eth).bool(), (eth).bool(), (1-eth).bool()]
save_path_list = ['', 'eth-cau', 'eth-oth']
for idx_list, save_path in zip(subset_list, save_path_list):
    new_path = os.path.join(path,save_path) # save results in separate folders
    make_dir(new_path)

    #### Evaluate and Save Results
    output_accy, output_auc, output_counter, output_f1, output_precision, output_recall, output_roc_curve = evaluate_LP(pred[idx_list], scores[idx_list], t_feature_labels[idx_list], t_aux_bank[idx_list, :], aux_col)
    metrics = ['accuracy', 'auc', 'counter', 'f1', 'precision', 'recall']
    output = [output_accy, output_auc, output_counter, output_f1, output_precision, output_recall]


    for tmp_df, metric in zip(output, metrics):
        tmp_df = append_experiment_info(tmp_df, model_path, feat_option, method, use_aux, metric)
        save_str = '_'.join([args.model_path, args.dataset, method, args.target, str(args.feat_option), str(use_aux), metric, args.val_idx_list,str(qc_filter_code)])
        filename = os.path.join(new_path, 'result_%s.pkl' % (save_str))
        print(filename)
        tmp_df.to_pickle(filename)

    # ROC Curve
    save_str = '_'.join([args.model_path, args.dataset, method, args.target, str(args.feat_option), str(use_aux), 'ROCCURVE', args.val_idx_list,str(qc_filter_code)])
    filename = os.path.join(new_path, 'result_%s.pkl' % (save_str))
    save_dict(output_roc_curve, filename)
    print(filename)
    print('done!')

