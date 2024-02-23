import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def arg_parse():
    parser = argparse.ArgumentParser(
        description='visualization')
    parser.add_argument('--gen_path', type=str,
                        help='data gen path')
    parser.add_argument('--orig_path', type=str,
                        help='data orig path')
    parser.add_argument('--save_path', type=str,
                        help='data saving dir')
    parser.set_defaults(
        gen_path='',
        orig_path='',
        save_path=''
    )
    return parser.parse_args()


def visual_trend_plot(data, title, dir):
    cluster = []
    for i in range(len(data)):
        plt.plot(data[i, :, 0], c="gray", alpha=0.4)
        cluster.append(data[i, :, 0])
    if len(cluster) > 0:
        plt.plot(np.average(np.vstack(cluster), axis=0), c="red")
    plt.title("Cluster")
    plt.savefig(dir + "/" + title + ".png")
    plt.show()
    plt.clf()


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


def MinMaxScaler(data):
    """Min Max normalizer.
    Args:
      - data: original data
    Returns:
      - norm_data: normalized data
    """
    #  min_val = np.min(np.min(data, axis=0), axis=0)
    #  data = data - min_val
    #  max_val = np.max(np.max(data, axis=0), axis=0)
    #  norm_data = data / (max_val - min_val + 1e-7)
    #  return norm_data, max_val, min_val
    min_data_temp = np.min(data, 1)
    max_data_temp = np.max(data, 1)
    min_data = np.reshape(
        np.repeat(min_data_temp, data.shape[1], axis=0), data.shape)
    max_data = np.reshape(
        np.repeat(max_data_temp, data.shape[1], axis=0), data.shape)
    numerator = data - min_data
    denominator = max_data - min_data
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, max_data_temp, min_data_temp


def time_gan_MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, numerator, denominator


def visual_basic_plot(x, y, x_name, y_name, title, dir):
    plt.plot(x, y, label=title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend()
    plt.savefig(dir + "/" + title + ".png")
    plt.show()
    plt.clf()


def visual_basic_gen_data(gen_data, real_data, num_day, title, dir):
    plt.plot(range(num_day), gen_data, label='gen')
    plt.plot(range(num_day), real_data, label='real')
    plt.xlabel('time')
    plt.ylabel('census')
    plt.title('ICU census')
    plt.legend()
    plt.savefig(dir + "/" + title + ".png")
    plt.clf()


def visualization(fake, real, option, dir):
    icu_fake, icu_real = fake[:, :, 0], real[:, :, 0]
    ms_fake, ms_real = fake[:, :, 1], real[:, :, 1]

    anal_sample_no = min([1000, len(real)])
    colors = ["red" for i in range(anal_sample_no)] + \
        ["blue" for i in range(anal_sample_no)]
    if option == 'pca':
        # PCA Analysis
        pca_icu = PCA(n_components=2)
        pca_icu.fit(icu_real)
        icu_real_res = pca_icu.transform(icu_real)
        icu_fake_res = pca_icu.transform(icu_fake)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(icu_real_res[:, 0], icu_real_res[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="ICU real")
        plt.scatter(icu_fake_res[:, 0], icu_fake_res[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="ICU fake")

        ax.legend()
        plt.title('ICU PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.savefig(dir + "/icu_pca.png")
        plt.show()
        plt.clf()

        pca_ms = PCA(n_components=2)
        pca_ms.fit(ms_real)
        ms_real_res = pca_ms.transform(ms_real)
        ms_fake_res = pca_ms.transform(ms_fake)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(ms_real_res[:, 0], ms_fake_res[:, 1],
                    c=colors[:anal_sample_no], label="MS real")
        plt.scatter(ms_real_res[:, 0], ms_fake_res[:, 1],
                    c=colors[anal_sample_no:], label="MS fake")

        ax.legend()
        plt.title('MS PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.savefig(dir + "/ms_pca.png")
        plt.show()
        plt.clf()

    elif option == 'tsne':

        # Do t-SNE Analysis together
        icu = np.concatenate((icu_real, icu_fake), axis=0)

        # TSNE anlaysis
        tsne_icu = TSNE(n_components=2, verbose=1, perplexity=25, init='pca')
        tsne_icu_results = tsne_icu.fit_transform(icu)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_icu_results[:anal_sample_no, 0], tsne_icu_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="ICU fake")
        plt.scatter(tsne_icu_results[anal_sample_no:, 0], tsne_icu_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="ICU real")

        ax.legend()

        plt.title('t-SNE ICU plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(dir + "/icu_tsne.png")
        plt.show()
        plt.clf()

        ms = np.concatenate((ms_real, ms_fake), axis=0)

        # TSNE anlaysis
        tsne_ms = TSNE(n_components=2, verbose=1, perplexity=25)
        tsne_ms_results = tsne_ms.fit_transform(ms)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_ms_results[:anal_sample_no, 0], tsne_ms_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="MS real")
        plt.scatter(tsne_ms_results[anal_sample_no:, 0], tsne_ms_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="MS fake")

        ax.legend()

        plt.title('t-SNE MS plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(dir + "/ms_tsne.png")
        plt.show()
        plt.clf()


def visualization_total(ori_data, generated_data, analysis, plot_dir, plot_name):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + \
        ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        xpbot = np.percentile(pca_hat_results[:, 0], 1)
        xptop = np.percentile(pca_hat_results[:, 0], 99)
        xpad = 0.2*(xptop - xpbot)
        xmin = xpbot - xpad
        xmax = xptop + xpad

        ypbot = np.percentile(pca_hat_results[:, 1], 1)
        yptop = np.percentile(pca_hat_results[:, 1], 99)
        ypad = 0.2*(yptop - ypbot)
        ymin = ypbot - ypad
        ymax = yptop + ypad
        plt.axis([xmin, xmax, ymin, ymax])
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.5, label="Original", s=100)
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.5, label="Generated", s=100)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.savefig(plot_dir + '/' + plot_name + ".png")
        plt.savefig(plot_dir + '/' + plot_name + ".pdf")
        plt.show()
        plt.clf()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=35)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.5, label="Original", s=100)
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.5, label="Generated", s=100)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        plt.savefig(plot_dir + '/' + plot_name + ".png")
        plt.savefig(plot_dir + '/' + plot_name + ".pdf")
        plt.show()
        plt.clf()


if __name__ == '__main__':
    args = arg_parse()
    gen_load = np.load(args.gen_path)
    real_load = np.load(args.orig_path)
    save_path = args.save_path

    visual_trend_plot(data=gen_load, dir=save_path, title='generated_data')
    visual_trend_plot(data=real_load, dir=save_path, title='real_data')

    print("the shape is the same")
    print(gen_load.shape == real_load.shape)

    mean_fake = np.mean(gen_load, axis=0)
    mean_real = np.mean(real_load, axis=0)
    diff_mean = np.abs(mean_real - mean_fake)
    var_fake = np.var(gen_load, axis=0)
    var_real = np.var(real_load, axis=0)
    diff_var = np.abs(var_real - var_fake)
    print(diff_mean)
    print(diff_var)
    # real_load_normal = np.load(save_path + 'normalize_generated_data.npy')

    visualization(
        fake=gen_load[:, :, :], real=real_load[:, :, :], option='tsne', dir=save_path)
    visualization_total(generated_data=gen_load[:, :, :], ori_data=real_load[:, :, :],
                        analysis='tsne', plot_dir=save_path, plot_name='training_total_tsne')
    visualization_total(generated_data=gen_load[:, :, :], ori_data=real_load[:, :, :],
                        analysis='pca', plot_dir=save_path, plot_name='training_total_pca')
    # visualization_total(generated_data=gen_load[:,:,:], ori_data=test_real_load_normalized[:,:,:], analysis='tsne', plot_dir= save_path, plot_name='test_total_tsne')

    # visual_basic_plot(range(gen_load.shape[1]), gen_load[10,:,0],  x_name='Days', y_name='Census', title='Fake_10', dir=save_path)

    shuffle_index = np.random.permutation(range(gen_load.shape[0]))
    gen_load_shuffled = gen_load[shuffle_index]
    real_load_shuffled = real_load[shuffle_index]
    # gen_load_shuffled_norm, _, _ = MinMaxScaler(gen_load_shuffled)
    for i in range(5):
        plt.plot(range(gen_load_shuffled.shape[1]), gen_load_shuffled[i, :, 1])
        plt.plot(
            range(gen_load_shuffled.shape[1]), real_load_shuffled[i, :, 1])
        plt.show()
        plt.clf()
    for i in range(10):
        plt.plot(range(gen_load_shuffled.shape[1]), gen_load_shuffled[i, :, 1])
    plt.savefig(save_path+'/fake_10_path.png')
    plt.title('MS')
    plt.show()
    plt.clf()
    for i in range(10):
        plt.plot(
            range(gen_load_shuffled.shape[1]), real_load_shuffled[i, :, 1])
    plt.savefig(save_path + '/real_5_path.png')
    plt.title('MS')
    plt.show()
    plt.clf()
    print("lol")

    # for day in range(gen_load.shape[1] - 24):
    #     plt.hist(real_load[:,day, 0], bins=30)  # density=False would make counts
    #     plt.ylabel('Probability')
    #     plt.xlabel('Data')
    #     plt.title('real histo' + str(day))
    #     plt.savefig(save_path + '/histogram_' + str(day) + '.png')
    #     plt.show()
    #     plt.clf()

    # for day in range(gen_load.shape[1] - 24):
    #     plt.hist(real_load_normal[:,day, 0], bins=30)  # density=False would make counts
    #     plt.ylabel('Probability')
    #     plt.xlabel('Data')
    #     plt.title('nomal histo' + str(day))
    #     plt.show()
    #     plt.clf()
