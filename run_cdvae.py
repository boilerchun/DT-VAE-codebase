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
    parser.add_argument('--opt_decay_rate', type=float,
                        help='Optimizer decay rate')
    parser.add_argument('--weight_decay', type=int,
                        help='Optimizer weight decay')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--recon_weight', type=float,
                        help='Weight on reconstruction loss')
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
        epochs=600,
        opt='adam',
        seed=7777,
        # ===========
        data_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/',
        data_name='stock_decreasing_increasing',
        experiment_name="CDT-VAE_stock_decreasing_increasing",
        parent_dir="./experiments",
        # ===========
        opt_scheduler='step',
        opt_decay_step=80,
        opt_decay_rate=0.8,
        weight_decay=0,
        recon_weight=1,
        lr=0.001)
    return parser.parse_args()


def train_model(args, dir):
    real_data_batch_loader, _, _, num_series_feature, _, _, _ = get_data_loader(
        data_dir=args.data_dir, dataset_name=args.data_name, batch_size=args.batch_size, args=args, diff_load_flag=True)
    print('num_series_feature: ' + str(num_series_feature))

    vae = models.CDTVAE(input_dim=num_series_feature, hidden_dim=args.hidden_dim,
                     noise_dim=args.noise_dim, output_dim=num_series_feature, args=args)
    vae = vae.to(device)

    dvae_scheduler, dvae_optimizer = utils.build_optimizer(
        args, vae.parameters())
    graph_reconstruction_loss = []
    for epoch in range(args.epochs):
        kl_div_loss_total = 0
        reconstruction_loss_total = 0
        num_batch = 0
        vae.train()
        for real_data_batch in real_data_batch_loader:
            dvae_optimizer.zero_grad()
            kl_div, _, reconstruction_loss = vae.forward(x=torch.concat((real_data_batch['feature'].to(device), real_data_batch['first_day'].to(device).detach()), dim=-1), real=real_data_batch['feature'].to(
                device), first_day=real_data_batch['first_day'].to(device).detach(), args=args)

            loss_total = kl_div + args.recon_weight * reconstruction_loss
            loss_total.backward()
            dvae_optimizer.step()

            kl_div_loss_total += kl_div.item()

            reconstruction_loss_total += reconstruction_loss.item()
            num_batch += 1
        graph_reconstruction_loss.append(reconstruction_loss_total / num_batch)
        print(str(epoch) + ', reconstruction_loss: ' + str(format(reconstruction_loss_total /
              num_batch, '.4f')) + ' kl_div: ' + str(format(kl_div_loss_total / num_batch, '.4f')))
        dvae_scheduler.step()
    model_path = dir + '/saved_model/'
    modelisdir = os.path.isdir(model_path)
    if not modelisdir:
        os.mkdir(model_path)
    torch.save(vae, model_path + 'vae_trained_model.pth')


def test(model_dir, num_series, log_dir, args):
    diff_data_batch_loader, series_len, num_series, _, _, _, first_day = get_data_loader(
        data_dir=args.data_dir, dataset_name=args.data_name, batch_size=args.batch_size, args=args)

    model = torch.load(model_dir + 'vae_trained_model.pth').to(device)
    noise_fake,_,_ = model.prior.prior_forward(x=diff_data_batch_loader.dataset.first_day.to(device))
    gen_data = None
    model.eval()
    with torch.no_grad():
        _, _, x_fake = model.decoder.forward(x=torch.concat(
                (noise_fake, diff_data_batch_loader.dataset.first_day.to(device)), dim=-1).to(device))
        gen_data = (
                x_fake + diff_data_batch_loader.dataset.first_day.to(device)).cpu().detach().numpy()
    real_data = diff_data_batch_loader.dataset.feature.cpu().detach().numpy()
    np.save(log_dir + '/generated_data.npy', gen_data)
    np.save(log_dir + '/real_data.npy', real_data + first_day)


def main():
    args = arg_parse()
    # ===========
    dir = args.experiment_name
    parent = args.parent_dir
    # ===========
    path = os.path.join(parent, dir)
    isdir = os.path.isdir(path)
    if not isdir:
        os.mkdir(path)

    sys.stdout = Logger(path)

    torch.random.manual_seed(seed=args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    print('====================================')
    print('args: ', args)
    print('====================================')
    print(
        f'[INFO] Using dataset: {args.data_name}, Using device: {device}, Using seed: {args.seed}')
    print('====================================')
    # dir = "vae_dann_threshold_with_att_moment_len_30_shuffled" + str(args.data_name) + "_batch_size_" + str(
    #     args.batch_size) + "_lr_" + str(
    #     args.lr) + "_decay_rate_" + str(args.opt_decay_rate) + "_epoch_" + str(args.epochs) + "_noise_dim_" + str(
    #     args.noise_dim) + "_hidden_dim_" + str(
    #     args.hidden_dim) + "_decoder_layer_" + str(args.decoder_layer) + "_encoder_layer_" + str(
    #     args.encoder_layer) + "_d_layer_2" + str(args.d_layer)
    # parent = "./experiments/smooth_covid_diff_qaunt"

    train_model(args, path)
    test(path + '/saved_model/', 500, path, args)


if __name__ == '__main__':
    main()
