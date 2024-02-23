import argparse
import utils

import torch
import models
import random
import matplotlib.pyplot as plt
import numpy as np
import os

import sys

from utils import get_data_loader, gen_fake_data, Logger
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def arg_parse():
    parser = argparse.ArgumentParser(description='VAE arguments.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--encoder_layer', type=int,
                        help='Number of encoder layers')
    parser.add_argument('--decoder_layer', type=int,
                        help='Number of decoder layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Number of hidden dimension')
    parser.add_argument('--noise_dim', type=int,
                        help='Gaussian noise dimension')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--opt', type=str,
                        help='Optimizer type')
    parser.add_argument('--seed', type=int,
                        help='Random seed')
    parser.add_argument('--data_dir', type=str,
                        help='Data directory')
    parser.add_argument('--data_name', type=str,
                        help='Data name')
    parser.add_argument('--opt_scheduler', type=str,
                        help='Optimizer scheduler type')
    parser.add_argument('--opt_decay_step', type=int,
                        help='Optimizer decay step')
    parser.add_argument('--opt_decay_rate', type=int,
                        help='Optimizer decay rate')
    parser.add_argument('--weight_decay', type=int,
                        help='Optimizer weight decay')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment name')
    parser.add_argument('--parent_dir', type=str,
                        help='Experiment Parent directory')
    parser.set_defaults(
        encoder_layer=2,
        decoder_layer=4,
        batch_size=50,
        hidden_dim=32,
        noise_dim=15,
        dropout=0,
        epochs=650,
        opt='adam',
        seed=7777,
        # ===========
        data_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/',
        data_name='stock_decreasing_increasing',
        experiment_name="T-VAE_stock_decreasing_increasing",
        parent_dir="./experiments",
        # ===========
        opt_scheduler='step',
        opt_decay_step=80,
        opt_decay_rate=0.8,
        weight_decay=0,
        lr=0.001)
    return parser.parse_args()


def train_model(args, dir):
    real_data_batch_loader, series_len, num_series, num_series_feature, _, _, _ = get_data_loader(
        data_dir=args.data_dir, dataset_name=args.data_name, batch_size=args.batch_size, args=args, diff_load_flag=False)
    vae = models.VAE(input_dim=num_series_feature, hidden_dim=args.hidden_dim, noise_dim=args.noise_dim,
                     output_dim=num_series_feature, args=args)

    vae = vae.to(device)

    vae_scheduler, vae_optimizer = utils.build_optimizer(
        args, vae.parameters())
    graph_reconstruction_loss = []
    for epoch in range(args.epochs):
        kl_div_loss_total = 0
        reconstruction_loss_total = 0

        num_batch = 0
        vae.train()
        for real_data_batch in real_data_batch_loader:
            vae_optimizer.zero_grad()
            kl_div, _, reconstruction_loss = vae.forward(x=real_data_batch['feature'].to(
                device), real=real_data_batch['feature'].to(device).detach(), args=args)

            loss_total = kl_div + reconstruction_loss

            loss_total.backward()
            vae_optimizer.step()

            kl_div_loss_total += kl_div.item()

            reconstruction_loss_total += reconstruction_loss.item()
            num_batch += 1
        graph_reconstruction_loss.append(reconstruction_loss_total / num_batch)
        print(
            str(epoch) + ', reconstruction_loss: ' + str(format(reconstruction_loss_total / num_batch, '.4f')) + ' kl_div: ' + str(format(kl_div_loss_total / num_batch, '.4f')))
        vae_scheduler.step()
    model_path = dir + '/saved_model/'
    modelisdir = os.path.isdir(model_path)
    if not modelisdir:
        os.mkdir(model_path)
    torch.save(vae, model_path + 'vae_trained_model.pth')


def test(model_dir, log_dir, args):
    diff_data_batch_loader, series_len, num_series, num_series_feature, data_max, data_min,_ = get_data_loader(
        data_dir=args.data_dir, dataset_name=args.data_name, batch_size=args.batch_size, args=args, diff_load_flag=False)
    noise_fake = gen_fake_data(data_len=num_series, dim=args.noise_dim, seq_len=series_len,
                               batch_size=num_series, diff_load_flag=False)
    print(noise_fake.dataset.feature)
    model = torch.load(model_dir + 'vae_trained_model.pth').to(device)
    # data_min = np.reshape(np.repeat(data_min, series_len, axis=0), (num_series, series_len, num_series_feature))
    # data_max = np.reshape(np.repeat(data_max, series_len, axis=0), (num_series, series_len, num_series_feature))
    # data_min_torch = torch.from_numpy(data_min).to(device)
    # data_max_torch = torch.from_numpy(data_max).to(device)
    gen_data = None
    for noise_fake_batch in noise_fake:
        model.eval()
        with torch.no_grad():
            _, _, x_fake = model.decoder.forward(
                x=noise_fake_batch['feature'].to(device))
            gen_data = x_fake.cpu().detach().numpy()
    real_data = diff_data_batch_loader.dataset.feature.cpu().detach().numpy()
    np.save(log_dir + '/generated_data.npy', gen_data)
    np.save(log_dir + '/real_data.npy', real_data)


def main():
    args = arg_parse()

    # dir = "Cluster_4"
    # parent = "./experiments/states_covid_long_term"
    dir = args.experiment_name
    parent = args.parent_dir
    path = os.path.join(parent, dir)
    isdir = os.path.isdir(path)
    if not isdir:
        os.mkdir(path)

    sys.stdout = Logger(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.random.manual_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print('====================================')
    print('args: ', args)
    print('====================================')
    print(
        f'[INFO] Using dataset: {args.data_name}, Using device: {device}, Using seed: {args.seed}')
    print('====================================')
    train_model(args, path)
    test(path + '/saved_model/', path, args)


if __name__ == '__main__':
    main()
