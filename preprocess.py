import numpy as np
import os


def shuffle_mix_data(cluster_1, cluster_2, random_seed=0):
    r"""
    Shuffle the mixed data.
    :param cluster_1: the data of cluster 1
    :param cluster_2: the data of cluster 2
    :param random_seed: the random seed
    :return: the shuffled mixed data
    """
    mix_data = np.concatenate((cluster_1, cluster_2), axis=0)
    data_len = mix_data.shape[0]
    np.random.seed(random_seed)
    idx_permute = np.random.permutation(range(0, data_len))
    mix_data_shuffled = mix_data[idx_permute, :, :]
    return mix_data_shuffled


def MixClusters(cluster_1_dir, cluster_2_dir, cluster_1_name, cluster_2_name, save_dir, random_seed=0):
    r"""
    Mix two clusters and save the mixed data to save_dir.
    :param cluster_1_dir: the directory of cluster 1
    :param cluster_2_dir: the directory of cluster 2
    :param cluster_1_name: the name of cluster 1
    :param cluster_2_name: the name of cluster 2
    :param save_dir: the directory to save the mixed data
    :param random_seed: the random seed
    :return: None
    """
    print("=====================================")
    print("Mixing clusters: " + cluster_1_name + " and " + cluster_2_name)
    cluster_1_train = np.load(cluster_1_dir + cluster_1_name + "_train.npy")
    cluster_1_test = np.load(cluster_1_dir + cluster_1_name + "_test.npy")

    cluster_2_train = np.load(cluster_2_dir + cluster_2_name + "_train.npy")
    cluster_2_test = np.load(cluster_2_dir + cluster_2_name + "_test.npy")

    mix_train = shuffle_mix_data(cluster_1_train, cluster_2_train, random_seed=random_seed)
    mix_test = shuffle_mix_data(cluster_1_test, cluster_2_test, random_seed=random_seed)

    save_dir = save_dir + cluster_1_name + "_" + cluster_2_name + "/"
    if not (os.path.isdir(save_dir)):
        os.makedirs(save_dir)
    np.save(save_dir + cluster_1_name + "_" + cluster_2_name + "_train.npy", mix_train)
    np.save(save_dir + cluster_1_name + "_" + cluster_2_name + "_test.npy", mix_test)
    print("Done!")
    print("=====================================")
    print()
