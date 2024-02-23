from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
import math
from sklearn.preprocessing import MinMaxScaler
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Cluster')
    parser.add_argument('--data_path', type=str,
                        help='data_path')
    parser.add_argument('--save_path', type=str,
                        help='save directory')
    parser.add_argument('--num_days', type=int,
                        help='num_days')
    parser.add_argument('--num_features', type=int,
                        help='num_features')
    parser.add_argument('--num_clusters', type=int,
                        help='num_clusters')
    parser.add_argument('--seed', type=int,
                        help='seed')
    parser.add_argument('--visual_dim', type=int,
                        help='visual_dim')
    parser.set_defaults(
        data_path="../../mix_cluster_generate_data/vae_diff_GAN_generated_data_cluster_mix_2_3.npy",
        save_path="../../mix_cluster_generate_data/vae_diff_GAN_generated_data_cluster_mix_2_3.png",
        num_days=49,
        num_features=2,
        num_clusters=2,
        seed=7,
        visual_dim=0
    )
    return parser.parse_args()


def get_clusters_and_visual( num_days, num_features, args, seed=7, num_clusters=2, visual_dim=0):
    print("---start clustering---")
    np.random.seed(seed)
    rst = np.load(args.data_path)
    rst = rst.reshape(-1, num_days, num_features)
    orig = rst.copy()
    for i in range(len(rst)):
        sc = MinMaxScaler()
        rst[i] = sc.fit_transform(rst[i])
    mySeries = rst.copy()
    cluster_count = num_clusters

    km = TimeSeriesKMeans(n_clusters=cluster_count,
                        metric="dtw", random_state=7)

    labels = km.fit_predict(mySeries)


    for label in list(sorted(set(labels))):
        plt.figure(figsize=(5, 5))
        cluster = []
        for i in range(len(labels)):
            if(labels[i] == label):
                cluster.append(mySeries[i, :, visual_dim])
        cluster_m = np.array(cluster)
        cluster_m = cluster_m[:len(cluster_m)//30*30, :]
        cluster_m = cluster_m.reshape((30, -1, 49))
        cluster_m = cluster_m.mean(axis=1)
        for i in range(len(cluster_m)):
            plt.plot(cluster_m[i], c="gray", alpha=0.4)
        if len(cluster) > 0:
            plt.plot(dtw_barycenter_averaging(np.vstack(cluster)), c="red")
        plt.savefig(args.save_path.split(".png")[0] + "_" + str(label)+".pdf")
        plt.show()
        plt.close()
    plt.show()

    fig, axs = plt.subplots(1, cluster_count, figsize=(12, 5))
    fig.suptitle('Clusters')
    column_j = 0
    for label in sorted(list(set(labels))):
        cluster = []
        for i in range(len(labels)):
            if(labels[i] == label):
                cluster.append(mySeries[i, :, visual_dim])
        cluster_m = np.array(cluster)
        cluster_m = cluster_m[:len(cluster_m)//30*30, :]
        cluster_m = cluster_m.reshape((30, -1, 49))
        cluster_m = cluster_m.mean(axis=1)
        print(cluster_m.shape)
        for i in range(len(cluster_m)):
            axs[column_j].plot(cluster_m[i], c="gray", alpha=0.4)
        if len(cluster) > 0:
            tmp = dtw_barycenter_averaging(np.vstack(cluster))
            axs[column_j].plot(tmp, c="red")
            np.save(args.save_path.split(".png")[
                    0] + f"_DBA_cluster_{label}.npy", tmp)
        axs[column_j].set_title("Cluster "+str(column_j))
        column_j += 1

    print("---finish clustering---")
    plt.savefig(args.save_path)
    plt.show()
    plt.clf()

def main():
    args = arg_parse()
    get_clusters_and_visual(num_days=args.num_days, num_features=args.num_features, args=args, seed=args.seed, num_clusters=args.num_clusters)


if __name__ == '__main__':
    main()
