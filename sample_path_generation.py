import torch
import argparse
import numpy as np
import random


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Downstream task synthetic generation')
    parser.add_argument('--noise_dim', type=int,
                        help='noise dimension for VAE')
    parser.add_argument('--save_dir', type=str,
                        help='save directory')
    parser.add_argument('--dataset_dir', type=str,
                        help='dataset_dir')
    parser.add_argument('--dataset_name', type=str,
                        help='dataset_name')
    parser.add_argument('--num_series_per_path', type=int,
                        help='num_series_per_path')
    parser.add_argument('--model_dir', type=str,
                        help='model_dir')
    parser.add_argument('--diff_first_day', type=bool,
                          help='whether to use the first day data as the first day of the generated data')
    parser.add_argument('--model_name', type=str,
                        help='model_name')
    parser.add_argument('--device', type=str,
                        help='GPU device')
    parser.set_defaults(
        noise_dim=15,
        save_dir='./experiments/T-VAE-GAN_stock_decreasing_increasing/T-VAE-GAN_stock_decreasing_increasing.npy',
        dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/',
        dataset_name='decreasing_increasing_test.npy',
        num_series_per_path=2,
        model_dir='./experiments/T-VAE-GAN_stock_decreasing_increasing/',
        diff_first_day=False,
        model_name='vae_gan_trained_model',
        device='cuda:1'
    )
    return parser.parse_args()


def random_gen(data_len, z_dim, seq_len):
    z = np.random.multivariate_normal(mean=np.zeros(
        z_dim), cov=np.identity(z_dim), size=(data_len, seq_len, ))
    return z


def get_first_day_data(data_dir, dataset_name, num_series_per_path):
    ori_data = np.load(data_dir + dataset_name)
    series_len = ori_data.shape[1]
    num_series_feature = ori_data.shape[-1]
    num_series = num_series_per_path * ori_data.shape[0]
    first_day_repeated = np.repeat(
        ori_data[:, 0, :], num_series_per_path, axis=0)
    first_day = np.repeat(np.expand_dims(
        first_day_repeated, axis=1), ori_data.shape[1], axis=1)
    return series_len, num_series_feature, num_series, first_day


def generate_sample_path(data_dir, data_name, model_dir, num_series_per_path, device, args):
    model = torch.load(
        model_dir + 'saved_model/' + args.model_name + '.pth', map_location=device)
    series_len, num_series_feature, num_series, first_day = get_first_day_data(
        data_dir=data_dir, dataset_name=data_name, num_series_per_path=num_series_per_path)
    model.eval()
    gen_data = None
    with torch.no_grad():
        print(args.diff_first_day)
        if args.diff_first_day:
            noise_np = random_gen(num_series, args.noise_dim, series_len-1)
            noise_temp = torch.concat(
                (torch.from_numpy(noise_np), torch.from_numpy(first_day[:, 1:, :])), dim=-1)
        else:
            noise_temp = torch.from_numpy(random_gen(num_series, args.noise_dim, series_len))

        print(noise_temp.shape)
        _, _, x_fake = model.decoder.forward(x=noise_temp.to(device).float())
        if args.diff_first_day:
            diff_first_day = torch.zeros(
                (x_fake.shape[0], 1, x_fake.shape[2])).to(device)
            diff_total = torch.concat((diff_first_day, x_fake), dim=1)
            # x_day_recovered = torch.relu(
            #     torch.from_numpy(first_day).to(device) + diff_total)
            x_day_recovered = torch.from_numpy(first_day).to(device) + diff_total
            gen_data = x_day_recovered
        else:
            gen_data = x_fake
    gen_data_np = gen_data.cpu().detach().numpy()
    np.save(args.save_dir, gen_data_np)


if __name__ == '__main__':
    args = arg_parse()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # print("diff: ")
    # print(args.diff_first_day)
    generate_sample_path(data_dir=args.dataset_dir, data_name=args.dataset_name,
                         num_series_per_path=args.num_series_per_path, model_dir=args.model_dir, device=torch.device(args.device), args=args)
