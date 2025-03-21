import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
from utils import *

def load_dataset(dataset, **kwargs):
    # wrapper for calling pytorch dataset class
    if dataset == 'ukbb':
        return Spiro_ukbb(**kwargs)
    elif dataset == 'copdgene':
        return Spiro_copdgene(**kwargs)


class Spiro_ukbb(Dataset):

    def __init__(self, max_length = None, data_path= './Files/3blows_with_max_rejection.csv', id_path = './Files/quality_train_test_ids/ukbb/', padding = True, target = 'ratio', transform = [0], idx_list = 'train', sample_size = None, eob_method = 'none', blow_filter = 'none', qc_filter_code = 12, addn_metrics = False, transform_p=0.5, source_time_interval = 10, downsample_factor = 5, feature_volume_interval = 50, append_transform_flag = False, site_list = [], **kwargs):
        """
        args:
            transform: list of integers representing transform. Each element of list is a different transform
            idx_list: which set of id's to select. 'train', 'val', 'test', 'train_val'
            eob_method: how to deal with end of blow values (volumes > fvc). none, zero, or max.
            blow filter: filter to train on only the best blows or on a qc code, instead of all blows. 'none', or 'best', or 'qc'
            qc_filter_code: if 'blow filter' = 'qc', filters data for only a specific qc code (for use in feature averaging evaluation)
            source_time_interval: the amount of time each feature in the sample respresents, in milliseconds.
            downsample_factor: amount of downsampling for each sample.
            feature_volume_interval: the amount of volume each feature in the sample represents (after flow-volume transform) in ml.
            append_transform_flag: if True, appends the transform flag to the end of the sample. Can only be used with single transformations. [1,0,0] volumetime; [0,1,0] flowtime; [0,0,1] flowvolume
            site_list (list): list of sites to filter on.
        """
        self.padding = padding
        self.max_length = max_length
        self.sample_blow = (0 in transform) # if we are randomly sampling blows per ID
        self.source_time_interval = source_time_interval
        self.downsample_factor = downsample_factor
        self.new_time_interval = self.source_time_interval * self.downsample_factor
        self.feature_volume_interval = feature_volume_interval
        self.append_transform_flag = append_transform_flag

        # flags for applying augmentations during validation/test
        self.transform = True # apply transformation (needs to be manually toggled)
        self.transform_p = transform_p  # probability of applying flow-time and flow-volume transforms, if applied
        # self.fixed_transform = 'none'

        if sample_size is not None:
            df = pd.read_csv(data_path, index_col=0, nrows = sample_size)  # read dataframe
        else:
            df = pd.read_csv(data_path, index_col=0)  # read dataframe

            # use specific IDs for train/test data
            filter_ids = None
            if idx_list == 'train':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'train_id_full.npy')), columns = ['ID'])
            elif idx_list == 'test':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'test_id_full.npy')), columns = ['ID'])
            elif idx_list == 'val':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'val_id_full.npy')), columns = ['ID'])
            elif idx_list == 'train_val':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'train_val_id_full.npy')), columns = ['ID'])
            elif idx_list == 'train_50k':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'train_id_50k.npy')), columns = ['ID'])
            elif idx_list == 'val_35k':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'val_id_50k.npy')), columns = ['ID'])

            # bootstrap indices
            elif idx_list[:9] == 'bootstrap':
                parse = idx_list.split('_')
                idx = int(parse[2])
                split = parse[1]

                bootstrap_dict = load_dict(os.path.join(id_path, 'bootstrap_dict_v2.pkl'))
                filter_ids = bootstrap_dict[idx][split]
                filter_ids = pd.DataFrame(filter_ids, columns = ['ID'])

            # filter on data split
            if filter_ids is not None:
                df = df.merge(filter_ids, how = 'inner', on = 'ID')
                # df = filter_ids.merge(df, how = 'left', on = 'ID')

        # filter on only the best blows, if specified by blow_filter
        if blow_filter == 'best':
            df = df[df['max_flag'] == 1]
            if 0 in transform: transform.remove(0) # cannot do sampling augmentation when using best blow only.
            self.sample_blow = False
            df = df.reset_index(drop = True)

        # filter on site, if specified by site_list
        if len(site_list)>0: df = df[df['site'].isin(site_list)]



        self.id_mapping = torch.from_numpy(df['ID'].values)
        self.id_list = torch.tensor(df['ID'].unique().tolist(), dtype = torch.int64) # list of IDs

        ###############################
        # auxillary data
        self.aux = True
        self.aux_col = [
            'ID',
            'rejection_reason',
            'fvc.best',
            'fev1.best',
            'fvc',
            'fev1',
            'sex',
            'standing_height',
            'age_at_recruitment',
            'EVERSMK',
            'PY',
            'smoking_status',
            'curSmoke',
            'max_flag',
            'USER_REJECTED',
            'TEST_DURATION',
            'COUGHING',
            'END_OF_TEST',
            'TIME_TO_PEF',
            'START_OF_TEST',
            'qc1',
            'qc2',
            'qc3',
            'qc4',
            'time_to_event',
            'event',
            #'COPD_combined',
            #'Asthma_combined',
            #'Emphysema_combined',
            #'Bronchitis_combined',
            'ethnicity_selfreported',
        ]
        self.data_aux = torch.tensor(df[self.aux_col].values, dtype = torch.float32)

        ##################################
        # Added for qc filter
        ##################################
        tmp = (1- self.data_aux[:,self.aux_col.index('USER_REJECTED')]).reshape(-1,1).int().numpy()  # 1 if not rejected
        aux_addn = pd.DataFrame(tmp, columns = ['qc0.1']).astype('int16')

        # qc0.2 Start of test
        tmp =  (1- self.data_aux[:,self.aux_col.index('START_OF_TEST')]).reshape(-1,1).int().numpy() # 1 if not rejected
        aux_addn['qc0.2'] = tmp
        aux_addn['qc0.2'] = aux_addn['qc0.2'] * aux_addn['qc0.1']

        # qc0.3 Time to PEF
        tmp =  (1- self.data_aux[:,self.aux_col.index('TIME_TO_PEF')]).reshape(-1,1).int().numpy() # 1 if not rejected
        aux_addn['qc0.3'] = tmp
        aux_addn['qc0.3'] = aux_addn['qc0.3'] * aux_addn['qc0.2']

        # qc0.4 Coughing
        tmp =  (1- self.data_aux[:,self.aux_col.index('COUGHING')]).reshape(-1,1).int().numpy() # 1 if not rejected
        aux_addn['qc0.4'] = tmp
        aux_addn['qc0.4'] = aux_addn['qc0.4'] * aux_addn['qc0.3']

        # append to aux_bank
        self.data_aux = torch.cat((self.data_aux, torch.tensor(aux_addn.values)), axis = 1)
        self.aux_col = self.aux_col + [
            'qc0.1',
            'qc0.2',
            'qc0.3',
            'qc0.4',
        ]

        if blow_filter == 'qc':

            # filter
            aux_idx = filter_qc(dataset = 'ukbb', data_aux = self.data_aux, aux_col = self.aux_col, qc_filter_code = qc_filter_code) # get indices that pass the qc filter code
            
            print('original_shape:')
            print(df.shape)
            df = df.filter(items = aux_idx.cpu().tolist(), axis = 0)
            print('filtered_shape:')
            print(df.shape)
            df = df.reset_index(drop = True)
            # self.aux_col = self.aux_col[:-4]
            # self.data_aux = self.data_aux[:,:-4]
            self.data_aux = self.data_aux[aux_idx,:]
            self.id_mapping = torch.from_numpy(df['ID'].values)
            self.id_list = torch.tensor(df['ID'].unique().tolist(), dtype = torch.int64) # list of IDs

        # if calculating additional comparison metrics (FEV2, FEV3, etc.)
        if addn_metrics:
            aux_addn_cols = [
                'fev2',
                'fev3',
                'fev6',
                'fef25',
                'fef75',
            ]
            aux_addn = torch.tensor(df[aux_addn_cols].values, dtype = torch.float32)
            self.data_aux = torch.cat((self.data_aux, aux_addn), dim = 1)
            self.aux_col = self.aux_col + aux_addn_cols

        ###############################
        # Spirometry Data
        #data_tmp = df.iloc[:, 14:2014]
        data_tmp = df[np.arange(2000).astype('str').tolist()]
        data_tmp = data_tmp.to_numpy(dtype = np.short)
        data_tmp = data_tmp[:, ::downsample_factor] # downscale to 50ms
        if data_tmp.shape[1] < max_length:
            # pad data if less than max_length
            data_tmp = np.concatenate((data_tmp, np.zeros((data_tmp.shape[0], max_length - data_tmp.shape[1]), dtype = np.short)), axis = 1)
        data_tmp = data_tmp[:, :max_length]
        if eob_method == 'zero':
            blow_fvc = np.max(data_tmp, axis = 1)
            blow_fvc = np.repeat(blow_fvc.reshape(-1,1), axis = 1, repeats = data_tmp.shape[1])
            blow_mask = (data_tmp <= blow_fvc)
            data_tmp = np.multiply(data_tmp, blow_mask)
            
        elif eob_method == 'max':
            blow_fvc = np.max(data_tmp, axis = 1)
            blow_fvc = np.repeat(blow_fvc.reshape(-1,1), axis = 1, repeats = data_tmp.shape[1])
            blow_mask = (data_tmp <= blow_fvc)
            data_tmp = np.multiply(data_tmp, blow_mask) 

            # set all values after fvc to be the fvc
            fvc_mask = (data_tmp == 0)
            fvc_mask[:,:10] = 0 # in case there are zeros at the beginning of blow
            data_tmp = data_tmp + np.multiply(fvc_mask, blow_fvc)

        self.data = torch.tensor(data_tmp, dtype = torch.float32)




        if target == 'ratio':
            self.target = torch.tensor(df['ratio.best'].values, dtype = torch.float32)
        elif target == 'fvc':
            self.target = torch.tensor(df['fvc.best'].values, dtype = torch.float32)
        elif target  == 'fev1':
            self.target = torch.tensor(df['fev1.best'].values, dtype = torch.float32)
        elif target  == 'binary_0.7_threshold':
            self.target = torch.tensor((df['ratio.best'].values<0.7)*1, dtype = torch.float32)
        elif target  == 'best_blow':
            self.target = torch.tensor(df['max_flag'].values, dtype = torch.float32)

        # transforms
        self.t_mem = [] # keep track of applied transforms
        self.t_list = []

        # if blow sampling is specified, this must be the first transformation
        if self.sample_blow:  
            self.t_mem.append(0)
            self.t_list.append(sample_blow)
        else:
            self.t_list.append(return_blow)

        # if using all of flow-time, flow-volume, and volume-time
        if (1 in transform) and (2 in transform):
            self.t_list.append(random_combine(volume_interval = self.feature_volume_interval, append_transform_flag = self.append_transform_flag))
            self.t_mem.append(1)
            self.t_mem.append(2)

        for t in transform:
            if t in self.t_mem:
                continue
            elif t == 1:
                self.t_list.append(random_flowtime(p = self.transform_p, append_transform_flag = self.append_transform_flag))
                self.t_mem.append(t)
            elif t == 2:
                self.t_list.append(random_flowvolume(p = self.transform_p, volume_interval = self.feature_volume_interval, append_transform_flag = self.append_transform_flag))
                self.t_mem.append(t)
            elif t == 3:
                self.t_list.append(random_mask())
                self.t_mem.append(t)
            elif t == 4:
                self.t_list.append(random_crop_mask())
                self.t_mem.append(t)
            elif t == 5:
                # flow-time and flow-volume
                self.t_list.append(random_combine(no_volumetime = True, append_transform_flag = self.append_transform_flag))
                self.t_mem.append(t)
        
        if (0 in self.t_mem and len(self.t_mem) == 1) or len(self.t_mem) == 0:
            # if no other transforms
            self.t_list.append(identity_transform(append_transform_flag=self.append_transform_flag))
            


    def __len__(self):
        if self.sample_blow:
            return len(self.id_list)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # self.transform can be turned on and off manually for validation accy
        if self.transform:
            for i, transform in enumerate(self.t_list):

                if i == 0:
                    # for blow sampling transformation
                    # sampled_id indicates the blow idx selected in random sample augmentation (list of 2)
                    x,y = transform(idx, dataset = self.data,ID_list = self.id_list, ID_mapping = self.id_mapping, target = self.target)

                    # ensure all blow are the same length
                    if self.padding:
                        for j, blow in enumerate(x):
                            if len(blow) < self.max_length:
                                blow = torch.cat((blow, torch.zeros(self.max_length - len(blow), dtype = torch.float32)))
                                x[j] = blow

                else:
                    x = transform(x)

        '''
        else:
            if 0 in self.t_mem:
                # for blow sampling transformation
                x,y = self.t_list[0](idx, dataset = self.data,ID_list = self.id_list, ID_mapping = self.id_mapping, target = self.target)

            x = self.data[idx,:]
            if self.fixed_transform == 'flowvolume':
                x = random_flowvolume(p=self.transform_p)(x).float()
            elif self.fixed_transform == 'flowtime':
                x = random_flowtime(p=self.transform_p)(x).float()
            elif self.fixed_transform == 'none':
                pass
            else:
                raise ValueError('Invalid fixed transform option, must be flowvolume, flowtime, or none')
            x = [x,x] # formatting 
            y = self.target[idx]
        '''

            
        if self.sample_blow:
            selected_ID = self.id_list[idx]  # ID of selected individual
            blow_idx = torch.where(self.id_mapping == selected_ID)[0] # index of selected individual's blows
            z = self.data_aux[blow_idx[0], :] # return the aux data of the first blow in for the ID.
        else:
            z = self.data_aux[idx,:]
        return x,y,z

class Spiro_copdgene(Dataset):

    def __init__(self, max_length = None, data_path= './Files/quality_copdgene_dataset.csv', id_path = './Files/quality_train_test_ids/copdgene/', padding = True, target = 'ratio', transform = [0], idx_list = 'train', sample_size = None, eob_method = 'none', blow_filter = 'none', qc_filter_code = 12, addn_metrics = False, transform_p=0.5, source_time_interval = 10, downsample_factor = 5, feature_volume_interval = 50, append_transform_flag = False, **kwargs):
        """
        args:
            transform: list of integers representing transform. Each element of list is a different transform
            idx_list: which set of id's to select. 'train', 'val', 'test', 'train_val'
            eob_method: how to deal with end of blow values (volumes > fvc). none, zero, or max.
            blow filter: filter to train on only the best blows or on a qc code, instead of all blows. 'none', or 'best', or 'qc'
            qc_filter_code: if 'blow filter' = 'qc', filters data for only a specific qc code (for use in feature averaging evaluation)
            source_time_interval: the amount of time each feature in the sample respresents, in milliseconds.
            downsample_factor: amount of downsampling for each sample.
            feature_volume_interval: the amount of volume each feature in the sample represents (after flow-volume transform) in ml.
            append_transform_flag: if True, appends the transform flag to the end of the sample. Can only be used with single transformations. [1,0,0] volumetime; [0,1,0] flowtime; [0,0,1] flowvolume
        """
        self.padding = padding
        self.max_length = max_length
        self.sample_blow = (0 in transform) # if we are randomly sampling blows per ID
        self.source_time_interval = source_time_interval
        self.downsample_factor = downsample_factor
        self.new_time_interval = self.source_time_interval * self.downsample_factor
        self.feature_volume_interval = feature_volume_interval
        self.append_transform_flag = append_transform_flag

        # flags for applying augmentations during validation/test
        self.transform = True # apply transformation (needs to be manually toggled)
        self.transform_p = transform_p  # probability of applying flow-time and flow-volume transforms, if applied
        # self.fixed_transform = 'none'

        if sample_size is not None:
            df = pd.read_csv(data_path, index_col=0, nrows = sample_size)  # read dataframe
        else:
            df = pd.read_csv(data_path, index_col=0)  # read dataframe

            # use specific IDs for train/test data
            if idx_list == 'train':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'train_id_full.npy')), columns = ['ID'])
            elif idx_list == 'test':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'test_id_full.npy')), columns = ['ID'])
            elif idx_list == 'val':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'val_id_full.npy')), columns = ['ID'])
            elif idx_list == 'train_val':
                filter_ids = pd.DataFrame(np.load(os.path.join(id_path, 'train_val_id_full.npy')), columns = ['ID'])

            elif idx_list[:9] == 'bootstrap':
                parse = idx_list.split('_')
                idx = int(parse[2])
                split = parse[1]

                bootstrap_dict = load_dict(os.path.join(id_path, 'bootstrap_dict_v2.pkl'))
                filter_ids = bootstrap_dict[idx][split]
                filter_ids = pd.DataFrame(filter_ids, columns = ['ID'])


            if idx_list != 'none':
                df = df.merge(filter_ids, how = 'inner', on = 'ID')
                # df = filter_ids.merge(df, how = 'left', on = 'ID')

        # filter on only the best blows, if specified by blow_filter
        if blow_filter == 'best':
            df = df[df['max_flag'] == 1]
            if 0 in transform: transform.remove(0) # cannot do sampling augmentation when using best blow only.
            df = df.reset_index(drop = True)

        self.id_mapping = torch.from_numpy(df['ID'].values)
        self.id_list = torch.tensor(df['ID'].unique().tolist(), dtype = torch.int64) # list of IDs
        

        ###############################
        # auxillary data
        self.aux = True
        self.aux_col = [
            'ID',
            'fvc.best',
            'fev1.best',
            'fvc',
            'fev1',
            'max_flag',
            'time_to_event',
            'event',
            'gold',
        ]
        self.data_aux = torch.tensor(df[self.aux_col].values, dtype = torch.float32)

        ##################################
        # Added for qc filter
        ##################################

        if blow_filter == 'qc':
            # filter
            aux_idx = filter_qc(dataset = 'copdgene', data_aux = self.data_aux, aux_col = self.aux_col, qc_filter_code = qc_filter_code) # get indices that pass the qc filter code
            
            
            print('original_shape:')
            print(df.shape)
            df = df.filter(items = aux_idx.cpu().tolist(), axis = 0)
            print('filtered_shape:')
            print(df.shape)
            df = df.reset_index(drop = True)
            self.data_aux = self.data_aux[aux_idx,:]
            self.id_mapping = torch.from_numpy(df['ID'].values)
            self.id_list = torch.tensor(df['ID'].unique().tolist(), dtype = torch.int64) # list of IDs


        # if calculating additional comparison metrics (FEV2, FEV3, etc.)
        if addn_metrics:
            aux_addn_cols = [
                'fev2',
                'fev3',
                'fev6',
                'fef25',
                'fef75',
            ]
            aux_addn = torch.tensor(df[aux_addn_cols].values, dtype = torch.float32)
            self.data_aux = torch.cat((self.data_aux, aux_addn), dim = 1)
            self.aux_col = self.aux_col + aux_addn_cols

        ###############################
        # Spirometry Data
        #data_tmp = df.iloc[:, 14:2014]
        data_tmp = df[np.arange(646).astype('str').tolist()]
        data_tmp = data_tmp.to_numpy(dtype = np.short)
        data_tmp = data_tmp[:, ::downsample_factor] # downscale. COPDGene is already at 60ms intervals
        data_tmp = data_tmp[:, :max_length]
        if eob_method == 'zero':
            blow_fvc = np.max(data_tmp, axis = 1)
            blow_fvc = np.repeat(blow_fvc.reshape(-1,1), axis = 1, repeats = data_tmp.shape[1])
            blow_mask = (data_tmp <= blow_fvc)
            data_tmp = np.multiply(data_tmp, blow_mask)
            
        elif eob_method == 'max':
            blow_fvc = np.max(data_tmp, axis = 1)
            blow_fvc = np.repeat(blow_fvc.reshape(-1,1), axis = 1, repeats = data_tmp.shape[1])
            blow_mask = (data_tmp <= blow_fvc)
            data_tmp = np.multiply(data_tmp, blow_mask) 

            # set all values after fvc to be the fvc
            fvc_mask = (data_tmp == 0)
            fvc_mask[:,:10] = 0 # in case there are zeros at the beginning of blow
            data_tmp = data_tmp + np.multiply(fvc_mask, blow_fvc)

        self.data = torch.tensor(data_tmp, dtype = torch.float32)
        
        if target == 'ratio':
            self.target = torch.tensor(df['ratio.best'].values, dtype = torch.float32)
        elif target == 'fvc':
            self.target = torch.tensor(df['fvc.best'].values, dtype = torch.float32)
        elif target  == 'fev1':
            self.target = torch.tensor(df['fev1.best'].values, dtype = torch.float32)
        elif target  == 'binary_0.7_threshold':
            self.target = torch.tensor((df['ratio.best'].values<0.7)*1, dtype = torch.float32)
        elif target  == 'best_blow':
            self.target = torch.tensor(df['max_flag'].values, dtype = torch.float32)

        # transforms
        self.t_mem = [] # keep track of applied transforms
        self.t_list = []

        # if blow sampling is specified, this must be the first transformation
        if self.sample_blow:  
            self.t_mem.append(0)
            self.t_list.append(sample_blow)
        else:
            self.t_list.append(return_blow)

        # if using all of flow-time, flow-volume, and volume-time
        if (1 in transform) and (2 in transform):
            self.t_list.append(random_combine(volume_interval = self.feature_volume_interval, append_transform_flag = self.append_transform_flag))
            self.t_mem.append(1)
            self.t_mem.append(2)

        for t in transform:
            if t in self.t_mem:
                continue
            elif t == 1:
                self.t_list.append(random_flowtime(p = self.transform_p, append_transform_flag = self.append_transform_flag))
                self.t_mem.append(t)
            elif t == 2:
                self.t_list.append(random_flowvolume(p = self.transform_p, volume_interval = self.feature_volume_interval, append_transform_flag = self.append_transform_flag))
                self.t_mem.append(t)
            elif t == 3:
                self.t_list.append(random_mask())
                self.t_mem.append(t)
            elif t == 4:
                self.t_list.append(random_crop_mask())
                self.t_mem.append(t)
            elif t == 5:
                # flow-time and flow-volume
                self.t_list.append(random_combine(no_volumetime = True, append_transform_flag = self.append_transform_flag))
                self.t_mem.append(t)
        
        if (0 in self.t_mem and len(self.t_mem) == 1) or len(self.t_mem) == 0:
            # if no other transforms
            self.t_list.append(identity_transform(append_transform_flag=self.append_transform_flag))
            


    def __len__(self):
        if self.sample_blow:
            return len(self.id_list)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # self.transform can be turned on and off manually for validation accy
        if self.transform:
            for i, transform in enumerate(self.t_list):

                if i == 0:
                    # for blow sampling transformation
                    # sampled_id indicates the blow idx selected in random sample augmentation (list of 2)
                    x,y = transform(idx, dataset = self.data,ID_list = self.id_list, ID_mapping = self.id_mapping, target = self.target)

                    # ensure all blow are the same length
                    if self.padding:
                        for j, blow in enumerate(x):
                            if len(blow) < self.max_length:
                                blow = torch.cat((blow, torch.zeros(self.max_length - len(blow), dtype = torch.float32)))
                                x[j] = blow

                else:
                    x = transform(x)

        '''
        else:
            if 0 in self.t_mem:
                # for blow sampling transformation
                x,y = self.t_list[0](idx, dataset = self.data,ID_list = self.id_list, ID_mapping = self.id_mapping, target = self.target)

            x = self.data[idx,:]
            if self.fixed_transform == 'flowvolume':
                x = random_flowvolume(p=self.transform_p)(x).float()
            elif self.fixed_transform == 'flowtime':
                x = random_flowtime(p=self.transform_p)(x).float()
            elif self.fixed_transform == 'none':
                pass
            else:
                raise ValueError('Invalid fixed transform option, must be flowvolume, flowtime, or none')
            x = [x,x] # formatting 
            y = self.target[idx]
        '''

            
        if self.sample_blow:
            selected_ID = self.id_list[idx]  # ID of selected individual
            blow_idx = torch.where(self.id_mapping == selected_ID)[0] # index of selected individual's blows
            z = self.data_aux[blow_idx[0], :] # return the aux data of the first blow in for the ID.
        else:
            z = self.data_aux[idx,:]
        return x,y,z