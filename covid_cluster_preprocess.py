import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

import os


def energy_MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def Stock_MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    # max_val = np.max(np.max(data, axis = 0), axis = 0)
    max_val = np.quantile(data, 0.6, axis=[0, 1])
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val


def split_train_test(data, train_ratio=0.5):
    r"""
    Split the time series data into training set and testing set.
    Parameters:
        data (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
        train_ratio (float): the ratio of the training set
    Returns:
        train (np.array): the training set
        test (np.array): the testing set
        """
    train = data[:int(data.shape[0] * train_ratio), :, :]
    test = data[int(data.shape[0] * train_ratio):, :, :]
    return train, test


def shuffle_data(data, random_seed=0):
    r"""
    Shuffle the time series data.
    Parameters:
        data (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
        random_seed (int): the random seed
    Returns:
        data (np.array): the shuffled time series data
        """
    np.random.seed(random_seed)
    idx_permute = np.random.permutation(data.shape[0])
    return data[idx_permute]


def getMinQuantile(data):
    r"""
    Get the minimum and quantile of the data.
    Parameters:
        data (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
    Returns:
        norm_data (np.array): the normalized time series data
        min_val (np.array): the minimum of the time series data
        quantile_val (np.array): the quantile of the time series data
    """
    min_val = np.min(data, axis=0)
    data = data - min_val

    quantile_val = np.quantile(data, 0.7, axis=0)
    norm_data = data / (quantile_val + 1e-7)

    return norm_data, min_val, quantile_val


def remove_zero(data, num_zero=30):
    r"""
    Remove the zero slices in the time series data.
    Parameters:
        data (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
    Returns:
        valid_slice_list (np.array): the time series data without zero slices
    """
    valid_slice_list = []
    for i in range(len(data)):
        non_zero_count = [np.count_nonzero(data[i, :, feature_idx]) > num_zero for feature_idx in range(data.shape[-1])]
        if all(non_zero_count):
            valid_slice_list.append(data[i, :, :])
    return np.array(valid_slice_list)


def weeklyMinQuantileScaler(data):
    r"""
    The weekly minimum and quantile scaler.
    Parameters:
        data (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
    Returns:
        reshape_data_processed (np.array): the weekly minimum and quantile scaled time series data
    """
    reshape_data = data.reshape(-1, 7, data.shape[-1])
    for i in range(7):
        reshape_data[:, i, :], _, _ = getMinQuantile(reshape_data[:, i, :])
    reshape_data_processed = reshape_data.reshape(-1, data.shape[1], data.shape[-1])
    return reshape_data_processed


def inverse_transform(min_val, max_val, data):
    r"""
    Inverse transform the data.
    Parameters:
        min_val (np.array): the minimum of the time series data
        max_val (np.array): the maximum of the time series data
        data (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
    Returns:
        data (np.array): the inverse transformed time series data
        """
    return data * (max_val + 1e-7) + min_val


def get_clusters(rst, n_clusters=5, metric="dtw", random_state=7):
    r"""
    Get clusters of time series data using DTW Barycenter Averaging.
    Parameters:
        rst (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
        n_clusters (int): the number of clusters
        metric (str): the metric used in DTW Barycenter Averaging
        random_state (int): the random state
    Returns:
        labels (:obj:np.array): the labels of the clusters
        transformed_rst (:obj:np.array): the transformed time series data"""
    transformed_rst = np.zeros(rst.shape)
    for i in range(len(rst)):
        sc = MinMaxScaler()
        transformed_rst[i] = sc.fit_transform(rst[i])
    # Use DTW Barycenter Averaging to get the barycenter
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state)
    labels = km.fit_predict(transformed_rst)
    return labels, transformed_rst

def get_dataset(rst, option, labels):
    r"""
    Get the dataset of clustered time series data.
    Parameters:
        rst (np.array): the original time series data (The shape is assumed to be (n_samples, time_steps, n_features)
        option (str): the option of the dataset, which can be "state" or "county"
        labels (:obj:np.array): the labels of the clusters
    Returns:
        """
    assert option in ["state", "county", "state_70", "county_70", "state_91", "county_91", "stock", "stock_70", "stock_91", "traffic", "traffic_70", "traffic_91", "energy", "energy_70", "energy_91"]
    save_dir = './datasets/Dataset_Dynamic_Trend_Preserve/'
    rst_dict = {}
    for i in range(len(rst)):
        rst_dict['cluster_'+str(labels[i])] = []
    
    # Create the directory to save the dataset
    if option == "state" or option == 'state_70' or option == 'state_91':
        if option == "state":
            save_dir = save_dir + 'State/'
        elif option == 'state_70':
            save_dir = save_dir + 'State_70/'
        elif option == 'state_91':
            save_dir = save_dir + 'State_91/'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
    elif option == "county" or option == 'county_70' or option == 'county_91':
        if option == "county":
            save_dir = save_dir + 'County/'
        elif option == 'county_70':
            save_dir = save_dir + 'County_70/'
        elif option == 'county_91':
            save_dir = save_dir + 'County_91/'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
    elif option == "stock" or option == 'stock_70' or option == 'stock_91':
        if option == "stock":
            save_dir = save_dir  + 'Stock_49/'
        elif option == 'stock_70':
            save_dir = save_dir + 'Stock_70/'
        elif option == 'stock_91':
            save_dir = save_dir + 'Stock_91/'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
    elif option == "traffic" or option == 'traffic_70' or option == 'traffic_91':
        if option == "traffic":
            save_dir = save_dir + 'traffic/'
        elif option == 'traffic_70':
            save_dir = save_dir + 'traffic_70/'
        elif option == 'traffic_91':
            save_dir = save_dir + 'traffic_91/'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
    elif option == "energy" or option == 'energy_70' or option == 'energy_91':
        if option == "energy":
            save_dir = save_dir + 'energy/'
        elif option == 'energy_70':
            save_dir = save_dir + 'energy_70/'
        elif option == 'energy_91':
            save_dir = save_dir + 'energy_91/'
    
    # Organize the clustered time series data into different clusters
    if option == "state":
        for i in range(len(rst)):
            rst_dict['cluster_'+str(labels[i])].append(rst[i])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_1'])
        np.save(save_dir+'early_peak.npy', rst_dict['cluster_2'])
        np.save(save_dir+'late_peak.npy', rst_dict['cluster_4'])
    elif option == "state_70":
        for i in range(len(rst)):
            # Here, we only use the first 70 days of the time series data, the reason of being not divided by 16 is that 
            if (i + 1) % 16 != 0:
                tmp = np.concatenate((rst[i, :, :], rst[i+1, :21, :]), axis=0)
                rst_dict['cluster_'+str(labels[i])].append(tmp)
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_1'])
        np.save(save_dir+'early_peak.npy', rst_dict['cluster_2'])
        np.save(save_dir+'late_peak.npy', rst_dict['cluster_4'])
    elif option == 'state_91':
        for i in range(len(rst)):
            if (i + 1) % 16 != 0:
                tmp = np.concatenate((rst[i, :, :], rst[i+1, :42, :]), axis=0)
                rst_dict['cluster_'+str(labels[i])].append(tmp) 
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_1'])
        np.save(save_dir+'early_peak.npy', rst_dict['cluster_2'])
        np.save(save_dir+'late_peak.npy', rst_dict['cluster_4'])
    elif option == 'county':
        for i in range(len(rst)):
            rst_dict['cluster_'+str(labels[i])].append(rst[i])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'through.npy', rst_dict['cluster_1'])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_3'])
        np.save(save_dir+'late_peak.npy', rst_dict['cluster_2'])
    elif option == 'county_70':
        for i in range(len(rst)):
            if (i + 1) % 16 != 0:
                tmp = np.concatenate((rst[i, :, :], rst[i+1, :21, :]), axis=0)
                rst_dict['cluster_'+str(labels[i])].append(tmp)
        np.save(save_dir+'increasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'through.npy', rst_dict['cluster_1'])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_3'])
        np.save(save_dir+'late_peak.npy', rst_dict['cluster_2'])
    elif option == 'county_91':
        for i in range(len(rst)):
            if (i + 1) % 16 != 0:
                tmp = np.concatenate((rst[i, :, :], rst[i+1, :42, :]), axis=0)
                rst_dict['cluster_'+str(labels[i])].append(tmp)
        np.save(save_dir+'increasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'through.npy', rst_dict['cluster_1'])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_3'])
        np.save(save_dir+'late_peak.npy', rst_dict['cluster_2'])
    elif option == 'stock':
        for i in range(len(rst)):
            rst_dict['cluster_'+str(labels[i])].append(rst[i])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_1'])
    elif option == 'stock_70':
        for i in range(len(rst)-3):
            tmp = np.concatenate((rst[i, :, :], rst[i+3, :21, :]), axis=0)
            rst_dict['cluster_'+str(labels[i])].append(tmp)
        np.save(save_dir+'increasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_1'])
    elif option == 'stock_91':
        for i in range(len(rst)-6):
            tmp = np.concatenate((rst[i, :, :], rst[i+6, :42, :]), axis=0)
            rst_dict['cluster_'+str(labels[i])].append(tmp)
        np.save(save_dir+'increasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_1'])
    elif option == 'traffic':
        for i in range(len(rst)):
            rst_dict['cluster_'+str(labels[i])].append(rst[i])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_0'])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_1'])
    elif option == 'traffic_70':
        orig = rst.copy()
        left = rst.copy()
        label_list = [1,2]
        orig_list = []
        for label in label_list:
            cluster_list = []
            for i in range(orig.shape[0]):
                if labels[i] == label:
                    if (i + 1) % 2 == 0:
                        ll = left[(i + 1)//2-1, :21].reshape(-1,1)
                        tmp = np.concatenate((orig[i, :, :], ll), axis=0)
                    else:
                        tmp = np.concatenate((orig[i, :, :], orig[i+1, :21, :]), axis=0)
                    cluster_list.append(tmp)
            cluster_list = np.array(cluster_list)
            orig_list.append(cluster_list)
    elif option == 'traffic_91':
        orig = rst.copy()
        left = rst.copy()
        label_list = [1,2]
        orig_list = []
        for label in label_list:
            cluster_list = []
            for i in range(orig.shape[0]):
                if labels[i] == label:
                    if (i + 1) % 2 == 0:
                        ll = left[(i + 1)//2-1, :42].reshape(-1,1)
                        tmp = np.concatenate((orig[i, :, :], ll), axis=0)
                    else:
                        tmp = np.concatenate((orig[i, :, :], orig[i+1, :42, :]), axis=0)
                    cluster_list.append(tmp)
            cluster_list = np.array(cluster_list)
            orig_list.append(cluster_list)

    elif option == 'energy':
        for i in range(len(rst)):
            rst_dict['cluster_'+str(labels[i])].append(rst[i])
        np.save(save_dir+'decreasing.npy', rst_dict['cluster_1'])
        np.save(save_dir+'increasing.npy', rst_dict['cluster_2'])
    elif option == 'energy_70':
        label_list = [0,1,2]
        orig_list = []
        orig = rst.copy()
        for label in label_list:
            cluster_list = []
            for i in range(orig.shape[0] - 1):
                if labels[i] == label:
                    tmp = np.concatenate((orig[i, :, :], orig[i+1, :21, :]), axis=0)
                    cluster_list.append(tmp)
            cluster_list = np.array(cluster_list)
            orig_list.append(cluster_list)
        np.save(save_dir+'decreasing_70.npy', np.array(orig_list[1]))
        np.save(save_dir+'increasing_70.npy', np.array(orig_list[2]))
    elif option == 'energy_91':
        label_list = [0,1,2]
        orig_list = []
        orig = rst.copy()
        for label in label_list:
            cluster_list = []
            for i in range(orig.shape[0] - 1):
                if labels[i] == label:
                    tmp = np.concatenate((orig[i, :, :], orig[i+1, :42, :]), axis=0)
                    cluster_list.append(tmp)
            cluster_list = np.array(cluster_list)
            orig_list.append(cluster_list)
        np.save(save_dir+'decreasing_91.npy', np.array(orig_list[1]))
        np.save(save_dir+'increasing_91.npy', np.array(orig_list[2]))


def raw_data_loading(raw_data_dir, option, n_clusters=5, metric="dtw", random_state=7, random_seed=7, num_days=784, time_step=49):
    r"""
    Load the raw data and cluster the raw data
    :param raw_data_dir: the directory of the raw data
    :param option: the option of the dataset
    :param n_clusters: the number of clusters
    :param metric: the metric used to calculate the distance between time series
    :param random_state: the random state
    :param random_seed: the random seed
    """
    np.random.seed(random_seed)
    print("===========================================")
    print("Loading the raw data...")
    if option == 'stock' or option == 'stock_70' or option == 'stock_91':
        raw_data_np = np.loadtxt(raw_data_dir, delimiter = ",",skiprows = 1)[:num_days, :]
    elif option == 'traffic':
        rst = []
        with open(raw_data_dir + "PEMS_train", "r") as f:
            for line in f:
                tmp = []
                l = line.strip()[1:-1].split(";")
                for i in l:
                    tmp.append(i.split(" "))
                rst.append(tmp)
        raw_data_np = np.array(rst).astype(np.float32)[0][:,:98]
    elif option == 'traffic_70' or option == 'traffic_91':
        rst = []
        with open(raw_data_dir + "PEMS_train", "r") as f:
            for line in f:
                tmp = []
                l = line.strip()[1:-1].split(";")
                for i in l:
                    tmp.append(i.split(" "))
                rst.append(tmp)
        raw_data_np = np.array(rst).astype(np.float32)[0][:,:144][:,-46:]
    elif option == 'energy' or option == 'energy_70' or option == 'energy_91':
        data = pd.read_csv(raw_data_dir + 'solar-AL.txt', header=None)
        data = data[1].to_numpy()
        raw_data_np = data[:52528].reshape(-1,49)
    else:
        raw_data_np = np.load(raw_data_dir)[:, :num_days, :]
    if option == 'stock' or option == 'stock_70' or option == 'stock_91':
        print("weekly skip sliding window")
        skip = 7
        temp_data = []
        for i in range(0, len(raw_data_np) - time_step + 1, skip):
            _x = raw_data_np[i:i + time_step]
            temp_data.append(_x)
        raw_data_np = np.array(temp_data)
    elif option == 'traffic' or option == 'traffic_70' or option == 'traffic_91':
        raw_data_np = raw_data_np.reshape(-1, time_step, 1)
    elif option == 'energy' or option == 'energy_70' or option == 'energy_91':
        raw_data_np = raw_data_np.reshape(-1, time_step, 1)
    else: 
        raw_data_np = raw_data_np.reshape(-1, time_step, raw_data_np.shape[-1])
    print("Clustering the raw data...")
    labels, _ = get_clusters(raw_data_np, n_clusters=n_clusters, metric=metric, random_state=random_state)
    print("Saving the clustered data...")
    get_dataset(raw_data_np, option, labels)
    print("Done!")
    print("===========================================")
    print()


def norm_and_split_train_test(data_dir, data_name, save_dir, train_ratio=0.5, random_seed=0, norm_option='weekly', remove_zero_flag=True, shuffle_flag=True, option=None, norm_flag=True):
    r"""
    Normalize the clustered data and split the data into training set and testing set
    :param data_dir: the directory to save the data
    :param data_name: the name of the data
    :param train_ratio: the ratio of the training set
    :param random_seed: the random seed
    :return: None
    """
    print("===========================================")
    clustered_data = np.load(data_dir+data_name+'.npy')
    print("Normalizing the clustered data...")
    if norm_flag:
        if norm_option == 'weekly':
            norm_data = weeklyMinQuantileScaler(clustered_data)
        elif norm_option == 'stock':
            norm_data,_,_ = Stock_MinMaxScaler(clustered_data)
        elif norm_option == 'energy':
            norm_data,_,_ = energy_MinMaxScaler(clustered_data)
    else:
        norm_data = clustered_data
    print("Splitting the data into training set and testing set...")
    if shuffle_flag:
        shuffle_norm_data = shuffle_data(norm_data, random_seed=random_seed)
    else:
        shuffle_norm_data = norm_data
    if remove_zero_flag:
        if option == 'stock' or option == 'stock_70' or option == 'stock_91':
            shuffle_norm_remove_zero_data = remove_zero(shuffle_norm_data, num_zero=4)
        else: 
            shuffle_norm_remove_zero_data = remove_zero(shuffle_norm_data)
    else:
        shuffle_norm_remove_zero_data = shuffle_norm_data
    train_data, test_data = split_train_test(shuffle_norm_remove_zero_data, train_ratio=train_ratio)
    np.save(save_dir+data_name+'_train.npy', train_data)
    np.save(save_dir+data_name+'_test.npy', test_data)
    print("Done!")
    print("===========================================")
    print()


def shuffle_mix_data(data_name_list, save_dir, data_save_name, random_seed=0, shuffle_flag=True):
    r"""
    Shuffle the mixed data
    :param data_name_list: the list of the data names
    :param save_dir: the directory to save the shuffled data
    :param random_seed: the random seed
    :return: None
    """
    counter = 0
    for data_name in data_name_list:
        if counter == 0:
            mix_data = np.load(save_dir+data_name+'.npy')
        else:
            mix_data = np.concatenate((mix_data, np.load(save_dir+data_name+'.npy')), axis=0)
        counter += 1
    if shuffle_flag:
        data_len = mix_data.shape[0]
        np.random.seed(random_seed)
        idx_permute = np.random.permutation(range(0, data_len))
        mix_data_shuffled = mix_data[idx_permute, :, :]
    else: 
        mix_data_shuffled = mix_data
    np.save(save_dir + data_save_name + '.npy', mix_data_shuffled)


def visualize_cluster_plots(labels, mySeries, visual_dim_idx, num_days, save_dir):
    for label in list(sorted(set(labels))):
        plt.figure(figsize=(5,5)) 
        cluster = []
        for i in range(len(labels)):
            if(labels[i]==label):
                cluster.append(mySeries[i,:, visual_dim_idx])
        cluster_m = np.array(cluster)
        print(cluster_m.shape)
        cluster_m = cluster_m[:len(cluster_m)//30*30,:]
        cluster_m = cluster_m.reshape((30,-1,num_days))
        cluster_m = cluster_m.mean(axis=1)
        for i in range(len(cluster_m)):
            plt.plot(cluster_m[i], c="gray", alpha=0.4)
        if len(cluster) > 0:
            plt.plot(dtw_barycenter_averaging(np.vstack(cluster)),c="red")
        # plt.savefig("cluster_state_icu_"+str(label)+".pdf")
        plt.show()
