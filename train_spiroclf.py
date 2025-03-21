import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

import time
import os
import sys
import io
from tqdm import tqdm
from argparse import ArgumentParser
import wandb
import matplotlib.pyplot as plt
import copy
import configargparse

from model_spiroclf import CNN
from load_data import load_dataset
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
from utils import *


# #ChanRev meth8

##########################################
# Parameters
##########################################

# Argument Source Priority (1 = Highest)
# 1) Command Line (if provided)
# 2) Config File (if provided)
# 3) Argparse Defaults

parser = configargparse.ArgParser(config_file_parser_class = configargparse.YAMLConfigFileParser)
parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

# Training
parser.add('--num_epochs', type=int, default=400)
parser.add('--initial_lr', type=float, default= 1e-2)
parser.add('--scheduler_gamma', type=float, default=0.1)
parser.add('--sample_size', type=int, default=-1, help = 'used for testing, reduces size of datafile')
parser.add('--resume_training_path', type=str, default=None, help = 'resume training from provided checkpoint file')
parser.add('--initialize_training_path', type=str, default=None, help = 'initialize model parameters from a given checkpoint file. training restarts at epoch 0.')
parser.add('--finetune', type=int, default=0, help = 'if 1, freeze all layers except fc2 for finetuning')
parser.add('--calc_test_accy', action='store_true', default=False, help = 'calculate test accuracy using knn')

# Model
parser.add('--model', type=str, default='cnn', help = 'rnn or cnn')
parser.add('--num_layers', type=int, default=6, help = 'number of convolution layers')
parser.add('--hidden_units', type=int, default=200, help = 'GRU hidden units')
parser.add('--kernel_size', type=int, default=40, help = 'kernel_size of CNN')
parser.add('--num_output_feat', type=int, default=128)
parser.add('--num_linear_units', type=int, default=100)
parser.add('--temperature', type=float, default=0.08)
parser.add('--weight_norm', action='store_true', default=False, help='boolean, apply weight normalization to convolutions')
parser.add('--conv_masking', action='store_true', default=False, help='boolean, mask convolutions to prevent access to future information')

# Dataset
parser.add('--dataset', type=str, default='ukbb', help = 'ukbb or copdgene')
parser.add('--files_path', type=str, default='./Files', help = 'path to files folder')
parser.add('--data_path', type=str, default='./Files/3blows_with_max_rejection.csv', help = 'path to files folder')
parser.add('--id_path', type=str, default='./Files/quality_train_test_ids/ukbb', help = 'path to files folder')
parser.add('--train_idx_list', type=str, default='train_50k', help = 'which set of IDs to use during training. "train", "train_val", "train_50k"')
parser.add('--val_idx_list', type=str, default='val_35k', help = 'which set of IDs to use during testing. "val", "test", "val_35k"')
parser.add('--blow_filter', type=str, default='none', help = '"none" or "best", filter to use only best blow during training')
parser.add('--eob_method', type=str, default='max', help = 'zero, max, or none')
parser.add('--target', type=str, default='binary_0.7_threshold')
parser.add('--batch_size', type=int, default=512)
parser.add('--source_time_interval', type=int, default=10)
parser.add('--data_downsample_factor', type=int, default=5)
parser.add('--feature_volume_interval', type=int, default=50)
parser.add('-t','--transform', action='append', help='define transformations')
parser.add('--transform_p', type=float, default=0.5, help='probability of flow-volume or flow-time transform')
parser.add('--dataloader_workers', type=int, default=0, help='Parameter num_workers for dataloader')
parser.add('--max_length', type=int, default=300, help='max length of sample')
parser.add('--append_transform_flag', type=int, default=0, help='1: append transform flag to samples, 0: do not append. Can only be used with single transformations. [1,0,0] volumetime; [0,1,0] flowtime; [0,0,1] flowvolume')
parser.add('-s','--site', action='append', help='if specified, filter on sites.')

# Tracking
parser.add('--verbose', action='store_true', default=False, help='boolean, use tqdm')
parser.add('--wandb_logging', action = 'store_true', default=False)

args = parser.parse_args()
print_args(args)

if not args.wandb_logging:
    os.environ['WANDB_MODE'] = 'dryrun'
else:
    os.environ['WANDB_MODE'] = 'online'


num_epochs = args.num_epochs
LR = args.initial_lr
batch_size = args.batch_size
hidden_units = args.hidden_units
num_layers = args.num_layers
kernel_size = args.kernel_size
verbose = args.verbose
temperature = args.temperature
max_length = args.max_length
if args.transform == None:
    transform_list = [0,1]
else:
    transform_list = [int(t) for t in args.transform]

if args.site == None:
    site_list = []
else:
    site_list = [int(t) for t in args.site]
if args.sample_size == -1:
    sample_size = None
else:
    sample_size = args.sample_size

# when using bootstrap samples, check that each training batch does not have duplicate IDs, which causes issues with contrastive loss
if (args.train_idx_list[:9] == 'bootstrap') and (0 in transform_list):
    check_batch_duplicates = True
else:
    check_batch_duplicates = False

model_name = '_'.join([str(t) for t in transform_list]) # save model name as list of transformations

np.random.seed(0)
torch.manual_seed(0)

if args.target == 'ratio':
    constraint = 'sigmoid'
    criterion = nn.MSELoss()
    calc_accy = False
elif args.target == 'binary_0.7_threshold':
    constraint = 'none'
    criterion = nn.BCEWithLogitsLoss()
    calc_accy = True
elif args.target == 'best_blow':
    constraint = 'none'
    criterion = nn.BCEWithLogitsLoss()
    calc_accy = True
else:
    constraint = 'relu'
    criterion = nn.MSELoss()
    calc_accy = False

files_path = args.files_path # path of the Files folder in repo
save_path = './models/spiroclf'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ' + str(device))
gpu_count = torch.cuda.device_count()
print('gpu count: ' + str(gpu_count))


##########################################
# Model
##########################################
if args.initialize_training_path is None:
    # initialize new model
    model = CNN(hidden_units, kernel_size = kernel_size, max_length = max_length, dropout_p=0, num_layers = num_layers, num_linear_units = args.num_linear_units, conv_masking = args.conv_masking, weight_norm = args.weight_norm, append_transform_flag=args.append_transform_flag)

    # Use multiple GPUs if available
    if gpu_count > 1 and torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids)
        batch_size *= gpu_count
        print('New Batch Size (scaled by gpu count): %s' % str(batch_size))
else:
    # initialize model parameters to the specified saved model
    checkpoint = torch.load(args.initialize_training_path, map_location = device)
    
    # load model hyperparameters
    model_params = checkpoint['args']
    model = CNN( model_params.hidden_units, kernel_size = model_params.kernel_size, max_length = model_params.max_length, dropout_p=0, num_layers = model_params.num_layers, num_linear_units = model_params.num_linear_units, conv_masking = model_params.conv_masking, weight_norm = model_params.weight_norm, append_transform_flag=args.append_transform_flag)
    model.to(device)

    if args.finetune:
        # if specified, freeze all layers except for the last fully-connected layer
        for name, param in model.named_parameters():
            if check_left_substring('fc2', name):
                param.requires_grad = True
            else:
                param.requires_grad = False 

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        # check if user is attempting to load nn.dataparallel checkpoint on non-parallel model
        model.load_state_dict(convert_dataparallel(checkpoint['model_state_dict']))

model.to(device)

##########################################
# Optimizer
##########################################
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr = 1e-6, verbose = verbose, patience = 20, factor = args.scheduler_gamma)


##########################################
# Load Checkpoint
##########################################
if args.resume_training_path is not None:
    # load checkpoint
    checkpoint = torch.load(args.resume_training_path, map_location = device)
    begin_epoch = checkpoint['epoch']
    run_name = checkpoint['run_name']
    run_id = checkpoint['run_id']
    test_val_min = checkpoint['test_val_min']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        # check if user is attempting to load nn.dataparallel checkpoint on non-parallel model
        model.load_state_dict(convert_dataparallel(checkpoint['model_state_dict']))

else:
    begin_epoch = 0 
    test_val_min = float('inf')


##########################################
# WandB Tracking
##########################################
if args.resume_training_path is None: run_id = wandb.util.generate_id() # generate unique ID for tracking
wandb.init(id = run_id, resume='allow')
wandb.config.update(args, allow_val_change = True)
wandb.watch(model)
if 'run_name' not in locals(): run_name = wandb.run.name
print('Model Name: %s' % run_name)


##########################################
# Dataset
##########################################
train_set = load_dataset(dataset = args.dataset, max_length = max_length,data_path = args.data_path,id_path = args.id_path, target = args.target, blow_filter = args.blow_filter, transform = transform_list, eob_method = args.eob_method, idx_list = args.train_idx_list, sample_size = sample_size, transform_p=args.transform_p, downsample_factor = args.data_downsample_factor, source_time_interval = args.source_time_interval, feature_volume_interval = args.feature_volume_interval, append_transform_flag = args.append_transform_flag, site_list = site_list)
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = args.dataloader_workers)

val_set = load_dataset(dataset = args.dataset, max_length = max_length,data_path = args.data_path,id_path = args.id_path, target = args.target, blow_filter = args.blow_filter, transform = transform_list, eob_method = args.eob_method, idx_list = args.val_idx_list, sample_size = sample_size, transform_p=args.transform_p, downsample_factor = args.data_downsample_factor, source_time_interval = args.source_time_interval, feature_volume_interval = args.feature_volume_interval, append_transform_flag = args.append_transform_flag, site_list = site_list)
val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = args.dataloader_workers)
print('train samples: ' + str(len(train_set)))
print('val samples: ' + str(len(val_set)))

if args.calc_test_accy: memory_loader = copy.deepcopy(train_loader)


##########################################
# Training
##########################################

optimizer.zero_grad()
test_acc_max = float('-inf')
num_batches = len(train_loader)

for epoch in range(begin_epoch, num_epochs):

    model.train()

    # trackers
    time1 = time.time()

    train_loss = train(model, train_loader, optimizer, temperature = temperature, verbose = verbose, epoch = epoch, check_batch_duplicates = check_batch_duplicates) 

    # epoch calculations
    time2 = time.time()
    time_train = time2 - time1

    # calculate test loss
    if epoch % 1 == 0 or (args.calc_test_accy and 'test_accy_1' not in locals()) or 'val_loss' not in locals():
        if args.calc_test_accy: test_acc_1, test_auc_1 = test_knn(model, memory_loader, val_loader, k=50)
        val_loss = test_loss(model, train_loader, temperature = temperature, verbose = False, check_batch_duplicates=check_batch_duplicates) 

        # Record best validation loss
        if val_loss < test_val_min:
            test_val_min = val_loss
            wandb.run.summary['val/loss'] = test_val_min
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'args': args,
                'run_name': run_name,
                'run_id': run_id,
                'test_val_min': test_val_min,
                }, os.path.join(files_path, save_path, 'simclr_%s_bestloss.pt' % run_name))
    # if test_acc_1 > test_acc_max:
    #     test_acc_max = test_acc_1
    #     wandb.run.summary['val/accy1'] = test_acc_max
    #     torch.save(model, os.path.join(save_path, 'simclr_%s_bestaccy.pt' % run_name))

    scheduler.step(val_loss)

    # Save Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
        'run_name': run_name,
        'run_id': run_id,
        'test_val_min': test_val_min,
        }, os.path.join(files_path, save_path, 'checkpoint_%s.pt' % run_name))

    # trackers
    time3 = time.time()
    time_epoch = time3 - time1

    # logging
    db = {}
    db['train/loss'] = train_loss
    db['train/epoch_time_train'] = time_train
    db['train/epoch_time_total'] = time_epoch
    db['train/lr'] = get_lr(optimizer)
    if args.calc_test_accy: db['val/accy1'] = test_acc_1
    db['val/loss'] = val_loss
    db['epoch'] = epoch
    wandb.log(db)

        
    print('=====================================')
    print("Epoch: %d/%d  || train_loss: %.4f  ||  Time: %f" % (
        epoch, num_epochs, train_loss, float(time.time()-time1)))
    sys.stdout.flush()
    


