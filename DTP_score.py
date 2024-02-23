from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw


def get_dba(seed, data_path, dim_option=0, cluster_count = 2):
    print(f"---Start Clustering with Seed {seed}---")
    np.random.seed(seed)
    rst = np.load(data_path)
    rst = rst.reshape(-1, rst.shape[-2], rst.shape[-1])
    for i in range(len(rst)):
        sc = MinMaxScaler()
        rst[i] = sc.fit_transform(rst[i])
    mySeries = rst.copy()
    km = TimeSeriesKMeans(n_clusters=cluster_count,
                          metric="dtw", random_state=seed)
    labels = km.fit_predict(mySeries)
    dba_list = []
    for label in sorted(list(set(labels))):
        cluster = []
        if dim_option == 'all':
            for i in range(len(labels)):
                if(labels[i] == label):
                    cluster.append(mySeries[i, :, :])
        else:
            for i in range(len(labels)):
                if(labels[i] == label):
                    cluster.append(mySeries[i, :, dim_option])
        if len(cluster) > 0:
            tmp = dtw_barycenter_averaging(np.vstack(cluster))
            dba_list.append(tmp)
    print(f"---Finish Clustering with Seed {seed}---")
    print()
    return dba_list


def get_orig_dba(data_path, num_day=49, dim_option=0):
    orig = np.load(data_path)
    orig = orig.reshape(-1, num_day, orig.shape[-1])
    for i in range(len(orig)):
        sc = MinMaxScaler()
        orig[i] = sc.fit_transform(orig[i])
    orig_tmp = orig.copy()
    cluster = []
    if dim_option == 'all':
        for i in range(len(orig_tmp)):
            cluster.append(orig_tmp[i, :, :])
    else:
        for i in range(len(orig_tmp)):
            cluster.append(orig_tmp[i, :, dim_option])
    orig_avg = dtw_barycenter_averaging(np.vstack(cluster))
    return orig_avg


def get_dtw_score(orig_c1_name, orig_c2_name, gen_mix_data, dim_option=0, seeds=[1, 2, 3, 4, 5]):
    orig_c1 = get_orig_dba(orig_c1_name, dim_option=dim_option)
    orig_c2 = get_orig_dba(orig_c2_name, dim_option=dim_option)
    rst_list = []
    for seed in seeds:
        # testing cluster 1 center
        # testing cluster 2 center
        # gen cluster 1 center
        # gen cluster 2 center
        dba_list = get_dba(seed, gen_mix_data, dim_option=dim_option)

        gen_c1 = dba_list[0]
        gen_c2 = dba_list[1]

        dtw_score1 = dtw(gen_c1, orig_c1)
        dtw_score2 = dtw(gen_c2, orig_c2)

        dtw_score3 = dtw(gen_c1, orig_c2)
        dtw_score4 = dtw(gen_c2, orig_c1)

        if dtw_score1+dtw_score2 > dtw_score3+dtw_score4:
            rst_list.append([dtw_score4, dtw_score3])
        else:
            rst_list.append([dtw_score1, dtw_score2])
    return rst_list

def get_dtp_results(orig_c1_name, orig_c2_name, gen_mix_data, dim_option=0, seeds=[1, 2, 3, 4, 5]):
    rst = get_dtw_score(orig_c1_name=orig_c1_name, orig_c2_name=orig_c2_name, gen_mix_data=gen_mix_data, dim_option=dim_option, seeds=seeds)
    rst = np.array(rst)
    m = np.mean(rst, axis=0)
    v = np.var(rst, axis=0)
    print(f"{m[0]:.4f}$\pm${v[0]*1:.5f}")
    print(f"{m[1]:.4f}$\pm${v[1]*1:.5f}")
    return m, v


