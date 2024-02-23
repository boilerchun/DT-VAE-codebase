import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys


class Logger(object):
    def __init__(self, dir):
        self.terminal = sys.stdout
        self.log = open(f"{dir}/log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class diff_Dataset(Dataset):
    def __init__(self, feature, first_day, series_len, num_series_feature, num_series):
        self.series_len = series_len
        self.num_series_feature = num_series_feature
        self.num_series = num_series
        self.feature = feature
        self.first_day = first_day

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        series_len = self.series_len
        num_series_feature = self.num_series_feature
        num_series = self.num_series
        data = self.feature[idx]
        first_day = self.first_day[idx]
        sample = {"feature": data, 'first_day': first_day, "series_len": series_len, "num_series_features": num_series_feature,
                  "num_series": num_series}
        return sample


class Dataset(Dataset):
    def __init__(self, feature, series_len, num_series_feature, num_series):
        self.series_len = series_len
        self.num_series_feature = num_series_feature
        self.num_series = num_series
        self.feature = feature

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        series_len = self.series_len
        num_series_feature = self.num_series_feature
        num_series = self.num_series
        data = self.feature[idx]
        sample = {"feature": data,  "series_len": series_len, "num_series_features": num_series_feature,
                  "num_series": num_series}
        return sample

def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val
    # min_data_temp = np.min(data, 1)
    # max_data_temp = np.max(data, 1)
    # min_data = np.reshape(np.repeat(min_data_temp, data.shape[1], axis=0), data.shape)
    # max_data = np.reshape(np.repeat(max_data_temp, data.shape[1], axis=0), data.shape)
    # numerator = data - min_data
    # denominator = max_data - min_data
    # norm_data = numerator / (denominator + 1e-7)
    # return norm_data, max_data_temp, min_data_temp


def weekly_MinMaxScaler(data):
    print("weekly norm")
    temp_data = []
    # Cut data by sequence length
    for i in range(0, data.shape[1] - 7, 7):
        _x = data[:, i:i + 7, :]
        temp_data.append(_x)
    temp_data_np = np.array(temp_data)
    weekly_max = np.swapaxes(np.max(temp_data_np, 2), 0, 1)
    weekly_min = np.swapaxes(np.min(temp_data_np, 2), 0, 1)
    weekly_max_repeat = np.repeat(weekly_max, 7, axis=1)
    weekly_min_repeat = np.repeat(weekly_min, 7, axis=1)

    weekly_data = data[:, :weekly_max_repeat.shape[1], :]
    numerator = weekly_data - weekly_min_repeat
    denominator = weekly_max_repeat - weekly_min_repeat
    norm_data = numerator / (denominator + 1e-7)
    print(norm_data.shape)
    return norm_data, weekly_max_repeat, weekly_min_repeat


def data_loading(data_dir, dataset_name, diff_load_flag):
    data_max, data_min = None, None
    if dataset_name == 'county_decreasing_and_late_peak':
        ori_data = np.load(
            data_dir + 'decreasing_late_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'county_decreasing_and_trough':
        ori_data = np.load(
            data_dir + 'decreasing_and_trough_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'county_increasing_and_late_peak':
        ori_data = np.load(
            data_dir + 'increasing_and_late_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'county_increasing_and_trough':
        ori_data = np.load(
            data_dir + 'increasing_through_train.npy')
        ori_data = ori_data[:, :, :]

    elif dataset_name == 'state_decreasing_and_late_peak':
        ori_data = np.load(
            data_dir + 'decreasing_late_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'state_decreasing_and_early_peak':
        ori_data = np.load(
            data_dir + 'decreasing_and_early_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'state_increasing_and_late_peak':
        ori_data = np.load(
            data_dir + 'increasing_and_late_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'state_increasing_and_early_peak':
        ori_data = np.load(
            data_dir + 'increasing_early_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_decreasing_increasing_late_peak':
        ori_data = np.load(
            data_dir + 'decreasing_increasing_late_peak_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'semi_syn_200':
        ori_data = np.load(
            data_dir + 'semi_syn_200.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'semi_syn_500':
        ori_data = np.load(
            data_dir + 'semi_syn_500.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'semi_syn_300':
        ori_data = np.load(
            data_dir + 'semi_syn_300.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'semi_syn_500_140_days':
        ori_data = np.load(
            data_dir + 'semi_syn_500_140_days.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'semi_syn_500_70_days':
        ori_data = np.load(
            data_dir + 'semi_syn_500_70_days_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'semi_syn_500_70_days_new':
        ori_data = np.load(
            data_dir + 'semi_syn_500_70_days_train_new.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'traffic_decreasing_and_trough':
        ori_data = np.load(
            data_dir + 'decreasing_and_trough_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'traffic_increasing_and_decreasing':
        ori_data = np.load(
            data_dir + 'increasing_decreasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'energy_increasing_and_decreasing':
        ori_data = np.load(
            data_dir + 'increasing_decreasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'simple_syn_gaussian':
        ori_data = np.load(
            data_dir + 'simple_syn_in_out_flow_no_flat.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'simple_syn_gaussian_high_var':
        ori_data = np.load(
            data_dir + 'simple_syn_in_out_flow_no_flat_high_var.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'simple_syn_gaussian_varing_var':
        ori_data = np.load(
            data_dir + 'simple_syn_in_out_flow_no_flat_varing_var.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'simple_syn_gaussian_fix_mean_var':
        ori_data = np.load(
            data_dir + 'simple_syn_fix_mean_var.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_decreasing_increasing_late_peak':
        ori_data = np.load(
            data_dir + 'decreasing_increasing_late_peak_test.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_decreasing_increasing':
        ori_data = np.load(
            data_dir + 'decreasing_increasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_decreasing':
        ori_data = np.load(
            data_dir + 'decreasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_increasing':
        ori_data = np.load(
            data_dir + 'increasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_70_decreasing':
        ori_data = np.load(
            data_dir + 'decreasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_70_increasing':
        ori_data = np.load(
            data_dir + 'increasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_91_decreasing':
        ori_data = np.load(
            data_dir + 'decreasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'stock_91_increasing':
        ori_data = np.load(
            data_dir + 'increasing_train.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'simple_syn_rw':
        ori_data = np.load(
            data_dir + 'simple_syn_gen_rw.npy')
        ori_data = ori_data[:, :, :]
    elif dataset_name == 'arr_dis_gaussian_simple_syn':
        ori_data = np.load(
            data_dir + 'simple_arr_dis_gaussian.npy')
        ori_data = ori_data[:, :, :]
    else:
        assert False, 'Unknown dataset_name: ' + dataset_name
    
    if diff_load_flag:
        first_day = np.repeat(np.expand_dims(
        ori_data[:, 0, :], axis=1), ori_data.shape[1] - 1, axis=1)
        ori_data = ori_data[:, 1:, :] - first_day
    else:
        first_day = None
    return ori_data, data_max, data_min, first_day


def stock_data_loader(data_dir, num_series):
    ori_data = pd.read_csv(data_dir + 'stock_data.csv')
    ori_data_np = ori_data.to_numpy()
    num_elements = int(len(ori_data) / num_series) * num_series
    ori_data_np = ori_data_np[:num_elements, :]
    num_features = ori_data_np.shape[1]
    ori_data_np = np.reshape(ori_data_np, (num_series, -1, num_features))
    return ori_data_np


def get_data_loader(data_dir, dataset_name, batch_size, args, diff_load_flag=True):
    data, data_max, data_min, first_day = data_loading(data_dir, dataset_name, diff_load_flag)

    series_len = data.shape[1]
    num_series_feature = data.shape[-1]
    num_series = data.shape[0]

    if diff_load_flag:
        diff_data_loader = diff_Dataset(feature=torch.from_numpy(data).float(), first_day=torch.from_numpy(first_day).float(),
                                 series_len=series_len, num_series_feature=num_series_feature, num_series=num_series,)
    else:
        diff_data_loader = Dataset(feature=torch.from_numpy(data).float(), series_len=series_len, num_series_feature=num_series_feature, num_series=num_series,)
    diff_data_batch = DataLoader(diff_data_loader, batch_size=batch_size)
    return diff_data_batch, series_len, num_series, num_series_feature, data_max, data_min, first_day


def random_gen(data_len, z_dim, seq_len):
    z = np.random.multivariate_normal(mean=np.zeros(
        z_dim), cov=np.identity(z_dim), size=(data_len, seq_len, ))
    # z = np.swapaxes(z, 1, -1)
    return z


def gen_fake_data(data_len, dim, seq_len, batch_size, first_day=None, diff_load_flag=True):
    fake = random_gen(data_len, dim, seq_len)
    if first_day is None:
        first_day = torch.zeros((data_len, dim, seq_len))
    if diff_load_flag:
        fake_loader = diff_Dataset(feature=torch.from_numpy(np.array(fake)).float(
        ), first_day=first_day, series_len=seq_len, num_series_feature=dim, num_series=data_len)
    else:
        fake_loader = Dataset(feature=torch.from_numpy(np.array(fake)).float(
        ), series_len=seq_len, num_series_feature=dim, num_series=data_len)
    fake_batch = DataLoader(fake_loader, batch_size=batch_size)
    return fake_batch
