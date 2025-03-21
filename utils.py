import torch
import numpy as np
import pandas as pd
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_auc_score
import multiprocessing
import torch.distributed as dist
import matplotlib.pyplot as plt
import wandb
import seaborn as sns


from simclr_loss_DDP import NT_Xent


def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    return x

def list2cuda(list):
    array = np.array(list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def auto2cuda(obj):
    # Checks object type, then calls corresponding function
    if type(obj) == list:
        return list2cuda(obj)
    elif type(obj) == np.ndarray:
        return numpy2cuda(obj)
    elif type(obj) == torch.Tensor:
        return tensor2cuda(obj)
    else:
        raise ValueError('input must be list, np array, or pytorch tensor')

def check_left_substring(substr, str):
    return str[:len(substr)] == substr

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return min(lr)

def freeze_layers(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_layers(model):
    for parameter in model.parameters():
        parameter.requires_grad = True

def load_model(path, model_path = None):
    if model_path is None:
        model_path = path

    tmp = os.path.dirname(os.path.abspath(path))
    sys.path.append(tmp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = torch.load(path, map_location=device)

    return model

def convert_dataparallel(state_dict):
    '''
    when wrapping a model with nn.DataParallel, the layers are prepended with 'module.'.
    This function 1) checks if layers are prepended with 'module.', then 2) removes this prepended layer name.
    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        prepend = k[:7]
        if prepend == 'module.':
            name = k[7:] # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

#========================================================
# io functions

import pickle
import os
import warnings

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def chdir_script():
    '''
    Changes current directory to that of the current python script
    '''
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def merge_dicts(dictionary_list):
    """
    merge a list of of dictionaries into a single dictionary
    """
    output = {}
    for dictionary in dictionary_list:
        output.update(dictionary)
    return output

def submit_slurm(python_script, job_file,job_out_dir = '', conda_env='a100', partition='gpu',mem=32, time_hrs = -1, n_gpu = 1, n_cpu = 4, exclude_nodes = None, job_name = 'script', prioritize_cpu_nodes = True, extra_line='', nodelist = None, gpu_type = 'v100-sxm2', requeue = False):
    '''
    submit batch job to slurm

    args:
        exclude_nodes: list of specific nodes to exclude
    '''
    if python_script is not None:
        dname = os.path.dirname(python_script.split(' -')[0]) # cut off script name before argparse options. This is to prevent issues when providing a path as a CLI argument.
    if job_out_dir == '':
        job_out = os.path.join(dname, 'job_out')
    else:
        job_out = os.path.join(job_out_dir, 'job_out')
    make_dir(job_out)  # create job_out folder

    if partition not in ['gpu', 'short', 'ai-jumpstart']:
        raise ValueError('invalid partition specified')

    # default time limits
    time_default = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':24
    }
    # max time limits
    time_max = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':48
    }
    if time_hrs == -1:
        # set to default time limit
        time_hrs = time_default[partition]
    elif time_hrs > time_max[partition]:
        # set to maximum time limit if exceeded
        time_hrs = time_max[partition]
        warnings.warn('time limit set to maximum for %s partiton: %s hours' % (partition, str(time_hrs)))
    elif time_hrs < 0:
        raise ValueError('invalid (negative) time specified')

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("\n")
        fh.writelines("#SBATCH --job-name=%s\n" % (job_name))
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --tasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=%s\n" % str(n_cpu))
        fh.writelines("#SBATCH --mem=%sGb \n" % str(mem))
        fh.writelines("#SBATCH --output=" + job_out + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + job_out + "/%j.err\n")
        fh.writelines("#SBATCH --partition=%s\n" % (partition))
        # fh.writelines("#SBATCH --nodelist=d3159")
        fh.writelines("#SBATCH --time=%s:00:00\n" % (str(time_hrs)))
        if nodelist is not None:
            fh.writelines("#SBATCH --nodelist=%s\n" % (nodelist))

        # exclude specific nodes
        if exclude_nodes is not None:
            exclude_str = ','.join(exclude_nodes)
            fh.writelines("#SBATCH --exclude=d[%s]\n" % (exclude_str))

        # specify gpu
        if partition == 'gpu':
            fh.writelines("#SBATCH --gres=gpu:%s:1\n" % gpu_type)
        elif partition == 'ai-jumpstart':
            if n_gpu>0:
                fh.writelines("#SBATCH --gres=gpu:a100:%s\n" % (str(n_gpu)))
            elif prioritize_cpu_nodes:
                # exclude gpu nodes
                fh.writelines("#SBATCH --exclude=d[3146-3150]\n")

        fh.writelines("\n")
        fh.writelines("CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh \n")
        fh.writelines("conda activate %s\n" % conda_env)
        fh.writelines("%s\n" % extra_line)
        if python_script is not None:
            if requeue:
                # end job early and requeue
                endtime = int(time_hrs * 60 - 10) # end job 10 min early
                fh.writelines("timeout %sm python -u %s\n" % (str(endtime), python_script))
                fh.writelines("if [[ $? == 124 ]]\n")
                fh.writelines("then\n")
                fh.writelines("    scontrol requeue $SLURM_JOB_ID\n")
                fh.writelines("fi\n")
            else:
                fh.writelines("python -u %s" % (python_script))
    os.system("sbatch %s" %job_file)


def print_args(args):
    for x, y in vars(args).items():
        print('{:<16} : {}'.format(x, y))

#========================================================
# np functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x>0)

def softplus(x):
    return np.log(1+np.exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def replace_nan(array, replacement_value = 0):
    array[np.isnan(array)] = replacement_value
    return array

#==========================================================
# Model Functions

def get_rank():
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def calc_test_loss_supervised(model, test_loader, criterion, calc_accy = False, aux_classes = 12):
    '''
    Calculate test loss

    args:
        model: pytorch model
        test_loader: dataloader object

    return:
        test loss
    '''
    model.eval()
    epoch_loss = 0.0
    num_batches = len(test_loader)
    correct = 0

    # track accuracy for each auxillary class
    aux_classes = 13
    aux_accy = tensor2cuda(torch.zeros(aux_classes))
    aux_loss = tensor2cuda(torch.zeros_like(aux_accy))
    aux_counter = tensor2cuda(torch.zeros_like(aux_accy))

    with torch.no_grad():

        for idx, ([data,_], target, aux) in enumerate(test_loader):
            data = tensor2cuda(data)
            target = tensor2cuda(target)
            aux = tensor2cuda(aux).reshape(-1,1)

            output = model(data, aux)
            loss = criterion(output, target)

            if calc_accy:
                pred = (output>=0)*1.
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                aux_onehot = tensor2cuda(torch.zeros((data.shape[0], aux_classes)))
                aux_onehot.scatter_(1,aux, 1)
                aux_counter = aux_counter + aux_onehot.sum(dim = 0)
                aux_accy = aux_accy + torch.matmul((pred.eq(target.view_as(pred))*1.), aux_onehot)
                #aux_loss = aux_loss + torch.matmul(loss.cpu(), aux_onehot).sum(dim = 0)

            epoch_loss += loss.item()

        epoch_loss /= num_batches
        if calc_accy:
            accy = (100.*correct/len(test_loader.dataset))
            aux_accy = torch.div(aux_accy, aux_counter)
        else:
            accy = -1
        return epoch_loss, accy, [aux_accy, aux_loss]

def simclr_loss(pos_1, pos_2, net, temperature, world_size):
    batch_size = pos_1.shape[0]
    pos_1, pos_2 = tensor2cuda(pos_1), tensor2cuda(pos_2)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2) # n x d
    if world_size >= 1:
        loss = NT_Xent(batch_size = batch_size, temperature = temperature, world_size = world_size)(out_1, out_2) # new loss for DDP
    else:
        out = torch.cat([out_1, out_2], dim=0) # 2n x d
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature) # 2n x 2n
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool() # 2n x 2n
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1) # 2n x (2n-1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


    return loss


def remove_duplicates(pos_1, pos_2, target, aux):
    ID_list = aux[:,0]
    unique_ID, counts = ID_list.unique(return_counts = True)
    duplicates = unique_ID[counts > 1]
    if len(duplicates) > 0:
        exclude_idx = []
        for duplicate in duplicates:
            exclude_idx.append(torch.where(ID_list == duplicate)[0][1:])
        exclude_idx = tensor2numpy(torch.tensor(exclude_idx))
        include_idx = numpy2cuda(np.setdiff1d(np.arange(len(ID_list)),exclude_idx))

        pos_1 = pos_1[include_idx,:]
        pos_2 = pos_2[include_idx,:]
        target = target[include_idx]
        aux = aux[include_idx,:]

    return pos_1, pos_2, target, aux

# adapted from https://github.com/leftthomas/SimCLR
def train(net, data_loader, train_optimizer, temperature = 0.5, verbose = True, epoch = 50, check_batch_duplicates = False, world_size = 1):
    net.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    if verbose:
        train_bar = tqdm(data_loader)
    else:
        train_bar = data_loader

    for [pos_1, pos_2], target, aux in train_bar:
        # if check_batch_duplicates:
        #     pos_1, pos_2, target, aux = remove_duplicates(pos_1, pos_2, target, aux)

        loss = simclr_loss(pos_1, pos_2, net, temperature, world_size) 
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_loss += loss.item() 
        
    # ## visualize sample
    if epoch < 10:
        n_samples = 5
        fig, ax = plt.subplots(1, n_samples, figsize=(20, 5))
        for i in range(n_samples):
            plot_data = tensor2numpy(torch.stack((pos_1[i,:], pos_2[i,:]), dim = 0))
            x_plot = np.arange(plot_data.shape[1])
            for j in range(2):
                sns.lineplot(x = x_plot, y = plot_data[j,:], ax =  ax[i])
        wandb.log({'blow visualization': wandb.Image(plt)})

    return total_loss / num_batches


def test_loss(net, data_loader, temperature = 0.5, verbose = True, check_batch_duplicates = False, world_size = 1):
    '''
    calculate test loss
    '''

    net.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    if verbose:
        train_bar = tqdm(data_loader)
    else:
        train_bar = data_loader

    with torch.no_grad():
        for [pos_1, pos_2], target, aux in train_bar:
            if check_batch_duplicates:
                pos_1, pos_2, target, aux = remove_duplicates(pos_1, pos_2, target, aux)
            loss = simclr_loss(pos_1, pos_2, net, temperature, world_size)
            total_loss += loss.item()
        
    return total_loss / num_batches


# adapted from https://github.com/leftthomas/SimCLR
# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_knn(net, memory_data_loader, test_data_loader, temperature = 0.1, k=200, transform = True, c = 2, eval_auc=False):
    net.eval()
    memory_data_loader.transform = transform
    test_data_loader.transform = transform

    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for [data, _], target, aux in memory_data_loader:
            data, target = tensor2cuda(data), tensor2cuda(target)
            feature, out = net(data)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        #test = torch.tensor(memory_data_loader.dataset.dataset.target, device=feature_bank.device)
        feature_labels = torch.cat(feature_labels).contiguous()
        # loop test data to predict the label by weighted knn search
        test_bar = test_data_loader

        for [data, _], target, aux in test_bar:
            data, target = tensor2cuda(data), tensor2cuda(target)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]

            # for each sample in test batch, this returns the labels for the 200 most similar samples in the training set.
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices).type(dtype=torch.int64)  # n x k
            sim_weight = (sim_weight / temperature).exp()  # n x k

            # counts for each class
            # c is number of classes, k is number of similar samples to compare labels (user-defined parameter)
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]

            # onehot encoding is applied to the similarity weights
            # i.e. the labels for the similar samples are weighted based on how similar they are to the test sample.
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            if eval_auc:
                auc = roc_auc_score(y_true = target.cpu(), y_score = torch.softmax(pred_scores, dim = 1)[:,1].cpu())
            else:
                auc = None

    return total_top1 / total_num * 100, auc


def extract_feats(model, dataloader, aux_only = False):
    '''
    aux_only: if True, skip running the model and only save aux_bank data
    '''
    feature_bank = []
    feature_labels = []
    aux_bank = []
    with torch.no_grad():
        # generate feature bank
        for [data, _], target, aux in dataloader:
            if aux_only:
                feature_bank.append(torch.zeros((5,5)))
            else:
                data = tensor2cuda(data)
                feature, out = model(data)
                #feature = torch.zeros((2,2))
                feature_bank.append(feature.cpu())
            feature_labels.append(target.cpu())
            aux_bank.append(aux.cpu())

    feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    feature_labels = torch.cat(feature_labels, dim=0).contiguous()
    aux_bank = torch.cat(aux_bank, dim = 0).contiguous()
    return feature_bank, feature_labels, aux_bank

def extract_feats_avg(model, dataloader, n_feat_input = 300, n_feat_output = 200, n_aug_samples = 20, batch_size = 1000):
    '''
    args:
        n_feat_input: size of input samples
        n_feat_output: size of learned representation
        n_aug_samples: number of times to sample augmentation
        batch_size: minibatch size
    '''

    dataset = dataloader.dataset
    n_ids = len(dataset) # number of participants to sample
    #n_ids = 500 # number of participants to sample
    n_batches = batch_size // n_aug_samples
    iterator = np.array_split(np.arange(n_ids), np.ceil(n_ids / n_batches))

    feature_bank = []
    feature_bank_1 = []
    feature_bank_2 = []
    feature_labels = []
    aux_bank = []

    from tqdm import tqdm
    with torch.no_grad():
        # generate feature bank
        for ID_batch in tqdm(iterator, total = len(iterator)):
            data_list = torch.zeros((len(ID_batch), 2, n_aug_samples // 2, n_feat_input))
            # dim 0: individual participant
            # dim 1: augmentation output (2 samples)
            # dim 2: number of specified augmentation iterations / 2 (since the dataloader outputs 2 augmentations per call)
            # dim 3: features
            for i, ID in enumerate(ID_batch): # Note that ID an index, not Participant ID
                for j in range(n_aug_samples // 2):
                    data, target, aux = dataset.__getitem__(ID)
                    data_list[i, 0, j, :] = data[0]
                    data_list[i, 1, j, :] = data[1]
                feature_labels.append(target)
                aux_bank.append(aux)

            data = data_list.reshape(-1,n_feat_input)
            #data = torch.stack(data_list, dim = 0)
            data = tensor2cuda(data)
            feature, out = model(data)
            feature = feature.reshape(-1, 2, n_aug_samples // 2, n_feat_output)
            for i in range(feature.shape[0]):
                tmp = feature[i,...].reshape(-1,feature.shape[-1])
                tmp = torch.unique(tmp, dim = 0)
                feature_bank.append(tmp.mean(dim = 0).cpu())
                #feature_bank_1.append(tmp[0,:])
                #feature_bank_2.append(tmp[1,:])
            #feature = feature.mean(dim = [1,2])
            #feature_bank.append(feature)

    feature_bank = torch.stack(feature_bank, dim=0).contiguous().cpu()
    #feature_bank_1 = torch.stack(feature_bank_1, dim=0).contiguous()
    #feature_bank_2 = torch.stack(feature_bank_2, dim=0).contiguous()
    feature_labels = torch.stack(feature_labels, dim=0).contiguous().cpu()
    aux_bank = torch.stack(aux_bank, dim = 0).contiguous().type(torch.float32).cpu()
    return feature_bank, feature_labels, aux_bank

#==========================================================
# Spirometry Functions
# from torchinterp1d import Interp1d
from scipy import interpolate

def calc_flowvolume(x, method = 'linear', fef_list = None, volume_interval = 50, append_transform_flag = False, **kwargs):
    '''
    uses scipy package; all vectors must by on cpu

    args:
        method: from interp1d, type of interpolation
        fef_list: optional, list of volumes for which to sample flow (for calculating fef). Otherwise returns full flowvolume curve.
        volume_interval: the amount of volume each feature should represent, in ml.
    '''

    if method != 'linear':
        x = tensor2numpy(x).reshape(-1)
        _,idx = np.unique(x, return_index = True)
        sorted_idx = np.sort(idx)
        x = x[sorted_idx]    
        x = torch.from_numpy(x.reshape(1,-1))

    flowtime = calc_flowtime(x, **kwargs)
    device = x.device
    x = x.float().cpu()
    y = flowtime.float().cpu()
    y = torch.relu(y) # apply relu to prevent issues with negative flow

    if fef_list is None:
        xnew = (torch.arange(0,10000, volume_interval)).cpu()
    else:
        xnew = torch.tensor(fef_list)

    with warnings.catch_warnings():
        # ignore invalid divide warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        # warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        # warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        # warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        f = interpolate.interp1d(x = x.reshape(-1), y = y.reshape(-1), kind = method, fill_value = 'extrapolate')
        output = f(xnew)
    output = np.nan_to_num(output, nan = 0.0).reshape(x.shape[0], -1)
    output = np.maximum(output, np.zeros_like(output))
    output = torch.from_numpy(output)

    # pad flowvolume curve to match input dimensions
    if fef_list is None:
        if output.shape[1] < x.shape[1]:
            output = torch.cat((output, (torch.zeros((1,x.shape[1] - output.shape[1])))), dim = 1)
        elif output.shape[1] > x.shape[1]:
            output = output[:,:x.shape[1]]
        output = output.float()

    if append_transform_flag:
        transform_flag = torch.tensor([0,0,1]).repeat(output.shape[0], 1)
        output = torch.cat((output, transform_flag), dim = 1)

    return output.to(device)

# def calc_flowvolume_gpu(x):
#     flowtime = calc_flowtime(x)
#     device = x.device
#     output = Interp1d()(x = x.float(), y = flowtime.float(), xnew = ((torch.arange(0,10000, 50)).to(device)))
#     if output.shape[1] < x.shape[1]:
#         output = torch.cat((output, (torch.zeros((1,x.shape[1] - output.shape[1]))).to(device)), dim = 1)
#     elif output.shape[1] > x.shape[1]:
#         output = output[:,:x.shape[1]]
#     return output.int()
    
def calc_flowtime(x, n_timepoints_avg = 1, interval = 60, reduce = False, append_transform_flag = False, **kwargs):
    '''
    convert volume-time to flow-time

    args:
        x: torch tensor of volume-time values
        n_timepoints_avg (int): number of timepoints to average accross
        interval (int): number of ms each timepoint represents
        reduce (bool): if true, divide output by (n_timepoints_avg * interval)
    
    '''
    x_shift = torch.cat((x[:, n_timepoints_avg:], x[:, -n_timepoints_avg:].reshape(-1, n_timepoints_avg)), dim = 1)
    # output = F.relu(x_shift - x) # output is in X ml / (n_timepoints_avg * interval) ms
    output = x_shift - x # output is in X ml / (n_timepoints_avg * interval) ms

    if reduce:
        output = output / (interval * n_timepoints_avg) # output is ml / 1 ms

    if append_transform_flag:
        transform_flag = torch.tensor([0,1,0]).repeat(output.shape[0], 1)
        output = torch.cat((output, transform_flag), dim = 1)

    return output

def calc_identity(x, append_transform_flag = False, **kwargs):
    '''
    identity transformation
    '''
    if append_transform_flag:
        if len(x.shape) == 1:
            transform_flag = torch.tensor([1,0,0])
            output = torch.cat((x, transform_flag), dim = 0)
        else:
            transform_flag = torch.tensor([1,0,0]).repeat(x.shape[0], 1)
            output = torch.cat((x, transform_flag), dim = 1)
        return output
    else:
        return x
    

def calc_pef(x, **kwargs):
    '''
    calculate pef

    args:
        x: torch vector of volume-time curve

    return:
        pef value (tensor)
        idx of pef (tensor)
        volume at pef index (tensor)
    '''

    flowtime = calc_flowtime(x, **kwargs)
    output = torch.max(flowtime, dim = 1)
    return output.values, output.indices, torch.gather(x, dim = 1, index = output.indices.reshape(-1,1)).reshape(-1)

def calc_pef_robust(x, threshold_pct = 0.1, **kwargs):
    '''
    returns the timepoint of the first instance of the flow value reaching within a threshold of the PEF

    args:
        x: torch vector of volume-time curve
        threshold_pct: threshold for calculating robust pef

    return:
        pef value (tensor)
        idx of pef (tensor)
        volume at pef index (tensor)
    '''
    flowtime = calc_flowtime(x, **kwargs)
    pef = flowtime.max(dim = 1).values
    threshold_value = (1-threshold_pct)* pef

    idx_list = []
    value_list = []
    volume_list = []
    for i in range(flowtime.shape[0]):
        idx = torch.where(flowtime[i,:]>= threshold_value[i])[0].min().item()
        value = flowtime[i,idx]
        volume = x[i,idx]
        idx_list.append(idx)
        value_list.append(value)
        volume_list.append(volume)
    return list2cuda(value_list), list2cuda(idx_list), list2cuda(volume_list)

def calc_fev1(x, interval = 50, **kwargs):
    '''
    Calculate FEV1
    
    args:
        x: data, x should be a numpy vector
        interval: milliseconds measured for each time point
    '''
    return calc_fevT(x, 1, interval, **kwargs)

def calc_fevT(x, T, interval = 10, return_pef = False, robust = False, n_timepoints_avg = 2, pad_fev1 = 0, **kwargs):
    '''
    Calculate FEV over T seconds.
    
    args:
        x: data, x should be a numpy vector
        T: number of seconds
        interval: milliseconds measured for each time point
        robust: use robust pef calculation
        pad_fev1: multiply final fev1 by 1+pad_fev1 for situations where timestep does not divide into 1 second evenly
    '''

    if type(x) == np.ndarray: x = numpy2cuda(x)
    if robust:
        threshold_pct = kwargs.get('threshold_pct', 0.1)
        pef, pef_ind, vol = calc_pef_robust(x, threshold_pct = threshold_pct, n_timepoints_avg = n_timepoints_avg, interval = interval, reduce = True)
    else:
        pef, pef_ind, vol = calc_pef(x, n_timepoints_avg = n_timepoints_avg, interval = interval, reduce = True)
    fev1 = []
    vol_timezero = []
    n_timepoints = (1000//interval) * T
    for i in range(x.shape[0]):
        
        # identify time zero
        y_intercept = vol[i] - pef[i] * (pef_ind[i] * interval) # y-intercept
        timezero = -y_intercept / pef[i]
        timezero_idx = int(max(np.round(timezero / interval, 0), 0)) # round to nearest positive timepoint

        # if pef_ind[i]+n_timepoints > (x.shape[1]-1): # if effort is not long enough, include volume before PEF
        #     exc = pef_ind[i]+n_timepoints - x.shape[1] + 1
        #     pef_ind[i] = pef_ind[i] - exc

        # ensure that calculated value represents the maximum attained volume (avoid issues with decreasing volume by making the volume-time curve monotonic)
        sample_maxvolume = np.maximum.accumulate(x[i,:])

        
        # in case time exceeds the length of the blow, set to the FVC value.
        if timezero_idx+n_timepoints >= (x.shape[1]):
            endpoint = sample_maxvolume[-1]
        else:
            endpoint = sample_maxvolume[timezero_idx+n_timepoints]

        fvc = sample_maxvolume.max()
        # if endpoint == 0: endpoint = fvc # if time exceeds blow length (and volume at T = 0), set to FVC
        
        output = (endpoint - sample_maxvolume[timezero_idx]) * (1+pad_fev1) # apply padding
        output = max(output, 0) # ensure positive value
        output = min(output, fvc) # Avoids issues with padding inflating fev1 values to be greater than fvc

        fev1.append(output)
        vol_timezero.append(x[i,timezero_idx])
    if return_pef:
        return list2cuda(fev1), pef, vol, timezero_idx, list2cuda(vol_timezero)
    else:
        return list2cuda(fev1)

def rolling_fevT(x, T, interval = 10, return_pef = False, pad_fev1 = 0, **kwargs):
    '''
    Calculate FEV over T seconds
    
    args:
        x: data, x should be a numpy vector
        T: number of seconds
        interval: milliseconds measured for each time point

    return:
        fev1 values
        pef index (if return_pef = True)
        volume at pef (if return_pef = True)

    '''

    if type(x) == np.ndarray: x = numpy2cuda(x)
    n_timepoints = (1000//interval) * T
    x_shift = torch.cat((x[:, n_timepoints:], x[:, -n_timepoints:].reshape(-1, n_timepoints)), dim = 1)
    output = (x_shift-x).max(axis = 1)
    if return_pef:
        return output.values * (1+pad_fev1), output.indices, torch.gather(x, dim = 1, index = output.indices.reshape(-1,1)).reshape(-1)
    else:
        return output.values * (1+pad_fev1)

def calc_fvc(x):
    '''
    Calculate FVC
    
    args:
        x: data, x should be a numpy vector
    '''
    return x.max(axis = 1)[0]

#==========================================================
# Transforms


class random_spiro_transform():
# parent class for spirometry transformations
    def __init__(self, n_processes = 1):
        self.n_processes = n_processes
        assert self.n_processes > 0

    # def single(self,x):
    #     # needs to be defined by child class
    #     return x
        
    def multiprocess(self, row_index):
        # wrapper for multiprocessing. Returns transformation for single row index. Assumes that self.x (matrix of samples) has been saved.
        x = self.x[row_index,...] # sample to transform
        return self.single(x)

    def __call__(self, x):
        '''
        args:
            x: x can be a torch vector or a list of such vectors
        '''
        if type(x) == list:
            if self.n_processes > 1:
                # use multiprocessing
                device = x[0].device
                self.x = torch.vstack(x).cpu()

                pool = multiprocessing.Pool(processes = self.n_processes)
                output = pool.map(self.multiprocess, range(len(x)))
                output = [sample.to(device) for sample in output] # move to GPU

            else:
                output = [self.single(sample) for sample in x]
            return output
        else:
            return self.single(x)


class random_flowvolume(random_spiro_transform):
    def __init__(self, p = 0.5, n_processes = 1, volume_interval = 50, append_transform_flag = False):
        super().__init__(n_processes = n_processes)
        self.p = p
        self.volume_interval = volume_interval
        self.append_transform_flag = append_transform_flag

    def single(self,x):
        # single random transformation
        r = np.random.binomial(n = 1, p = self.p)
        if r == 1:
            flowvolume = calc_flowvolume(x.unsqueeze(0), volume_interval = self.volume_interval, append_transform_flag = self.append_transform_flag).squeeze()
        
            #flowvolume *= 5  # scale to help model training
            return flowvolume
        else:
            return calc_identity(x, append_transform_flag = self.append_transform_flag)

class random_flowtime(random_spiro_transform):
    def __init__(self, p = 0.5, n_processes = 1, append_transform_flag = False):
        super().__init__(n_processes = n_processes)
        self.p = p
        self.append_transform_flag = append_transform_flag

    def single(self,x):
        # single random transformation
        r = np.random.binomial(n = 1, p = self.p)
        if r == 1:
            flowtime = calc_flowtime(x.unsqueeze(0), append_transform_flag = self.append_transform_flag).squeeze()
            #flowtime *= 5 # scale to help training
            return flowtime
        else:
            return calc_identity(x, append_transform_flag = self.append_transform_flag)

class identity_transform(random_spiro_transform):
    def __init__(self, n_processes = 1, append_transform_flag = False):
        super().__init__(n_processes = n_processes)
        self.append_transform_flag = append_transform_flag
    def single(self,x):
        return calc_identity(x, append_transform_flag = self.append_transform_flag)
    

class random_combine(random_spiro_transform):
    '''
    combine flowtime and flowvolume transform
    '''
    def __init__(self, no_volumetime = False, n_processes = 1, volume_interval = 50, append_transform_flag = False, **kwargs):
        super().__init__(n_processes = n_processes)
        self.no_volumetime = no_volumetime # flag to only use flowtime and flowvolume transforms
        self.volume_interval = volume_interval
        self.append_transform_flag = append_transform_flag

    def single(self,x):
        
        if self.no_volumetime: 
            select_high = 2
        else:
            select_high = 3

        r = np.random.randint(low = 0, high = select_high)
        if r == 0:
            output = calc_flowtime(x.unsqueeze(0), append_transform_flag = self.append_transform_flag).squeeze()
            #flowtime *= 5 # scale to help training
        elif r == 1:
            output = calc_flowvolume(x.unsqueeze(0), volume_interval = self.volume_interval, append_transform_flag = self.append_transform_flag).squeeze()
        else:
            output = calc_identity(x, append_transform_flag = self.append_transform_flag)
        return output

class random_crop_mask(random_spiro_transform):
    def __init__(self, low = 60, n_processes = 1):
        '''
        similar to random crop below, but only masks instead of reducing sample length
        '''
        super().__init__(n_processes = n_processes)
        self.low = low # minimum length to keep

    def single(self, x):
        '''
        args:
            x: torch vector
        '''
        crop_size = np.random.randint(low = self.low, high = len(x))
        x[crop_size:] = 0

        return x 

    
class random_crop(random_spiro_transform):
    def __init__(self, crop_size = 200, n_processes = 1):
        '''
        returns a random crop of the given sample
        '''
        super().__init__(n_processes = n_processes)
        self.crop_size = crop_size
    def single(self, x):
        '''
        args:
            x: torch vector
        '''
        if len(x)-self.crop_size <=0: import pdb; pdb.set_trace()
        attempts = 10
        for i in range(attempts):
            idx = np.random.randint(len(x)-self.crop_size)
            crop = x[idx:(idx+self.crop_size)]
            if len(x.nonzero()) > self.crop_size/5: break # ensure that at least 20% of the crop contains data

        return crop 
    
class random_mask(random_spiro_transform):
    def __init__(self, max_masks = 3, max_size = 30, n_processes = 1):
        '''
        randomly masks the data sample, with random number of masks of random size

        args:
            max_masks: maximum number of masks to apply
            max_size: maximum size of each mask
        '''
        super().__init__(n_processes = n_processes)
        self.max_masks = max_masks
        self.max_size = max_size

    def single(self, x):
        '''
        args:
            x: torch vector
        '''

        num_masks = np.random.randint(self.max_masks)
        if num_masks>0:
            mask_size = np.random.randint(self.max_size, size = num_masks)
            idx = np.random.randint(len(x) - mask_size.max(), size = num_masks)

            for i in range(num_masks):
                idx_begin = idx[i]
                idx_end = idx_begin + mask_size[i]
                x[idx_begin:idx_end] = 0
            
        return x 

def sample_blow(idx, dataset, ID_list, ID_mapping, target):
    '''
    This class takes an ID (integer) and returns a blow

    '''

    sample_list = []
    selected_ID = ID_list[idx]  # ID of selected individual
    blow_idx = torch.where(ID_mapping == selected_ID)[0] # index of selected individual's blows
    for i in blow_idx:
        sample_list.append(dataset[i,:])

    if len(blow_idx) > 2:
        # if there are more than 2 blows, filter out one
        # if len(blow_idx)>3: warnings.warn('More than 3 blows found for a given ID; possible duplicates')
        for j in range(len(blow_idx)-2):
            r = np.random.randint(len(sample_list))
            sample_list.pop(r)

    elif len(blow_idx) == 0:
        raise ValueError('Given ID has 0 blows')

    elif len(blow_idx) == 1:
        # return same blow twice.
        #warnings.warn('Given ID only has 1 blow')
        sample_list = sample_list + sample_list

    return sample_list, target[blow_idx[0]]


def return_blow(idx, dataset, target, **kwargs):
    '''
    returns a blow for given index (no transform)
    '''

    return [dataset[idx,:], dataset[idx,:]], target[idx]


def filter_qc(dataset, data_aux, aux_col, qc_filter_code):
    '''
    filters dataset based on qc_filter_code.

    returns indices in data_aux that pass the qc_filter_code
    '''

    if dataset == 'ukbb':
        if qc_filter_code ==0:
            aux_idx = torch.where(data_aux[:,aux_col.index('qc0.1')] == 0)[0]
        elif qc_filter_code ==1:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc0.2')] == 0) * (data_aux[:,aux_col.index('qc0.1')] == 1))[0]
        elif qc_filter_code ==2:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc0.3')] == 0) * (data_aux[:,aux_col.index('qc0.2')] == 1))[0]
        elif qc_filter_code ==3:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc0.4')] == 0) * (data_aux[:,aux_col.index('qc0.3')] == 1))[0]
        elif qc_filter_code ==4:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc1')] == 0) * (data_aux[:,aux_col.index('qc0.4')] == 1))[0]
        elif qc_filter_code ==5:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc2')] == 0) * (data_aux[:,aux_col.index('qc1')] == 1))[0]
        elif qc_filter_code ==6:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc3')] == 0) * (data_aux[:,aux_col.index('qc2')] == 1))[0]
        elif qc_filter_code ==7:
            aux_idx = torch.where((data_aux[:,aux_col.index('qc4')] == 0) * (data_aux[:,aux_col.index('qc3')] == 1))[0]
        elif qc_filter_code ==8:
            aux_idx = torch.where((data_aux[:,aux_col.index('max_flag')] == 0) * (data_aux[:,aux_col.index('qc4')] == 1))[0] # passed qc but not best blow (non-maximal)
        elif qc_filter_code ==9:
            aux_idx = torch.where(data_aux[:,aux_col.index('max_flag')] == 1)[0] # best blow
        elif qc_filter_code ==10:
            aux_idx = torch.where(data_aux[:,aux_col.index('max_flag')] == 0)[0] # all excluding best blow
        elif qc_filter_code ==11:
            aux_idx = torch.where(data_aux[:,aux_col.index('qc4')] == 1)[0] # all nonrejected
        elif qc_filter_code ==12:
            aux_idx = torch.arange(data_aux.shape[0]) # all blows
        else:
            raise ValueError('Invalid qc_filter_code for ukbb dataset')

    elif dataset == 'copdgene':
        if qc_filter_code ==9:
            aux_idx = torch.where(data_aux[:,aux_col.index('max_flag')] == 1)[0] # best blow
        elif qc_filter_code ==10:
            aux_idx = torch.where(data_aux[:,aux_col.index('max_flag')] == 0)[0] # all excluding best blow
        elif qc_filter_code ==11:
            aux_idx = torch.where(data_aux[:,aux_col.index('max_flag')] == 0)[0] # all nonrejected (In copdgene this is currently the same as qc_filter_code == 10)
        elif qc_filter_code ==12:
            aux_idx = torch.arange(data_aux.shape[0]) # all blows
        else:
            raise ValueError('Invalid qc_filter_code for copdgene dataset')

    else:
        raise ValueError('Invalid dataset')

    return aux_idx

def copdgene_lookup_covar(aux_bank, aux_col, cggrid, copdgene_id_mapping):
    '''
    return covariate matrix, labels, and features based on cggrid lookup

    incl_covar: boolean, if True, include cggrid covariates in covariate matrix
    incl_metrics: boolean, if True, include fev1/fvc/ratio in covariate matrix
    '''
    # get covar matrix
    colnames = ['ID']
    cols = [aux_col.index(col_name) for col_name in colnames]
    covar_matrix = pd.DataFrame(aux_bank[:,cols], columns = colnames)
    covar_matrix = covar_matrix.merge(copdgene_id_mapping, how = 'left', on = 'ID')
    covar_matrix['ID'] = covar_matrix['ID'].astype('long')

    covar = ['sid', 'Height_CM', 'age_visit', 'gender', 'race','FEV1pp_post']
    tmp = cggrid.sort_values(by = 'Phase_study').drop_duplicates('sid') # take values from earlier phase if available
    tmp = tmp[covar] # select covariates of interest
    covar_matrix = covar_matrix.merge(tmp, how = 'left', on = 'sid').reset_index(drop = True)
    return covar_matrix



def copdgene_fev1pp(aux_bank, aux_col, cggrid, copdgene_id_mapping):
    covar_matrix = copdgene_lookup_covar(aux_bank, aux_col, cggrid, copdgene_id_mapping)
    eth = covar_matrix.loc[:,['race']]
    race_map = {1: 'Cau',2: 'other'}
    eth['race'] = eth['race'].astype('category').cat.rename_categories(race_map)

    sex = covar_matrix.loc[:,['gender']]
    sex_map = {
        2: 'female',
        1: 'male'
    }
    sex['gender'] = sex['gender'].astype('category').cat.rename_categories(sex_map)

    from spiref import gli12
    from tqdm import tqdm
    rvc = gli12.GLIReferenceValueCalculator()

    fev1_pred = []
    for i in tqdm(range(aux_bank.shape[0])):
        fev1 = rvc.calculate_fev1(
            sex.iloc[i,0],
            covar_matrix.loc[i, 'Height_CM'],
            covar_matrix.loc[i, 'age_visit'],
            eth.iloc[i,0]
        )

        fev1_pred.append(fev1)

    fev1 = aux_bank[:,aux_col.index('fev1.best')].numpy()
    fev1_pred = np.array(fev1_pred)
    fev1_pp = 100*(fev1 / fev1_pred)
    return fev1_pp

def ukbb_fev1pp(aux_bank, aux_col):
    eth = pd.DataFrame(aux_bank[:,aux_col.index('ethnicity_selfreported')])
    eth[0] = eth[0].astype('str')
    eth.iloc[eth[0].str[:1]!='1', 0] = 'other'
    eth.iloc[eth[0].str[:1]=='1', 0] = 'Cau'

    sex = pd.DataFrame(aux_bank[:,aux_col.index('sex')])
    sex_map = {
        0: 'female',
        1: 'male'
    }
    sex = sex.replace({0:sex_map})

    from spiref import gli12
    from tqdm import tqdm
    rvc = gli12.GLIReferenceValueCalculator()

    fev1_pred = []
    for i in tqdm(range(aux_bank.shape[0])):
        fev1 = rvc.calculate_fev1(
            sex.iloc[i,0],
            aux_bank[:,aux_col.index('standing_height')].numpy()[i],
            aux_bank[:,aux_col.index('age_at_recruitment')].numpy()[i],
            eth.iloc[i,0]
        )
        fev1_pred.append(fev1)

    fev1 = aux_bank[:,aux_col.index('fev1.best')].numpy()
    fev1_pred = np.array(fev1_pred)
    fev1_pp = 100*(fev1 / fev1_pred)
    return fev1_pp