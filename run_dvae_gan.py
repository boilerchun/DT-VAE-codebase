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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def arg_parse():
    parser = argparse.ArgumentParser(description='VAE arguments.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--encoder_layer', type=int,
                        help='Number of encoder layers')
    parser.add_argument('--decoder_layer', type=int,
                        help='Number of decoder layers')
    parser.add_argument('--d_layer', type=int,
                        help='Number of discriminator layers')
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
    parser.add_argument('--ad_loss_weight', type=int,
                        help='Adversarial loss weight')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment name')
    parser.add_argument('--parent_dir', type=str,
                        help='Experiment Parent directory')
    parser.add_argument('--dis_recon_loss_weight', type=float,
                        help='dis_recon_loss_weight')
    parser.add_argument('--recon_loss_weight', type=float,
                        help='recon_loss_weight')
    parser.set_defaults(
        encoder_layer=3,
        decoder_layer=4,
        d_layer=2,
        batch_size=50,
        hidden_dim=46,
        noise_dim=15,
        dropout=0,
        epochs=500,
        opt='adam',
        seed=7777,
        # ========
        data_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/',
        data_name='stock_decreasing_increasing',
        experiment_name="DT-VAE-GAN_stock_decreasing_increasing",
        parent_dir="./experiments",
        # ========
        opt_scheduler='step',
        opt_decay_step=80,
        opt_decay_rate=0.8,
        weight_decay=0,
        lr=0.001,
        ad_loss_weight=15,  # small
        recon_loss_weight=1,
        dis_recon_loss_weight=0.2)
    return parser.parse_args()


def train_model(args, dir):
    real_data_batch_loader, series_len, num_series, num_series_feature, _, _, _ = get_data_loader(
        data_dir=args.data_dir, dataset_name=args.data_name, batch_size=args.batch_size, args=args, diff_load_flag=True)
    vae_gan = models.DTVAE_GAN(input_dim=num_series_feature, hidden_dim=args.hidden_dim, noise_dim=args.noise_dim,
                             output_dim=num_series_feature, args=args)

    vae_gan = vae_gan.to(device)

    vae_gan_encoder_scheduler, vae_gan_encoder_optimizer = utils.build_optimizer(
        args, vae_gan.encoder.parameters())
    vae_gan_dis_decoder_scheduler, vae_gan_dis_decoder_optimizer = utils.build_optimizer(
        args, tuple(vae_gan.decoder.parameters()) + tuple(vae_gan.discriminator.parameters()))

    graph_reconstruction_loss = []
    for epoch in range(args.epochs):
        noise_fake = gen_fake_data(
            data_len=num_series, dim=args.noise_dim, seq_len=series_len, batch_size=args.batch_size)
        kl_div_loss_total = 0
        reconstruction_loss_total = 0
        bce_loss_total = 0

        num_batch = 0
        vae_gan.train()
        for real_data_batch, noise_fake_data_batch in zip(real_data_batch_loader, noise_fake):
            vae_gan_encoder_optimizer.zero_grad()
            vae_gan_dis_decoder_optimizer.zero_grad()
            alpha = epoch / args.epochs * args.ad_loss_weight
            kl_div, _, reconstruction_loss, dis_reconstruct_loss, bce_loss = vae_gan.forward(x=real_data_batch['feature'].to(device), real=real_data_batch['feature'].to(
                device), first_day=real_data_batch['first_day'].to(device).detach(), fake=noise_fake_data_batch['feature'].to(device), alpha=alpha, args=args)
            decoder_dis_loss = bce_loss + reconstruction_loss + \
                args.dis_recon_loss_weight * dis_reconstruct_loss
            encoder_loss = args.recon_loss_weight * kl_div + reconstruction_loss + \
                args.dis_recon_loss_weight * dis_reconstruct_loss

            decoder_dis_loss.backward(inputs=tuple(vae_gan.decoder.parameters(
            )) + tuple(vae_gan.discriminator.parameters()), retain_graph=True)
            encoder_loss.backward(inputs=tuple(vae_gan.encoder.parameters()))
            vae_gan_dis_decoder_optimizer.step()
            vae_gan_encoder_optimizer.step()

            kl_div_loss_total += kl_div.item()
            bce_loss_total += bce_loss.item()
            reconstruction_loss_total += reconstruction_loss.item()
            num_batch += 1
        graph_reconstruction_loss.append(reconstruction_loss_total / num_batch)
        print(
            str(epoch) + ', reconstruction_loss: ' + str(format(reconstruction_loss_total / num_batch, '.4f')) + ' kl_div: ' + str(format(kl_div_loss_total / num_batch, '.4f')) + ', bce loss: ' + str(
                format(bce_loss_total / num_batch, '.4f')))
        vae_gan_encoder_scheduler.step()
        vae_gan_dis_decoder_scheduler.step()
    model_path = dir + '/saved_model/'
    modelisdir = os.path.isdir(model_path)
    if not modelisdir:
        os.mkdir(model_path)
    torch.save(vae_gan, model_path + 'vae_gan_trained_model.pth')


def test(model_dir, log_dir, args):
    diff_data_batch_loader, series_len, num_series, num_series_feature, data_max, data_min, first_day = get_data_loader(
        data_dir=args.data_dir, dataset_name=args.data_name, batch_size=args.batch_size, args=args)
    noise_fake = gen_fake_data(data_len=num_series, dim=args.noise_dim, seq_len=series_len,
                               batch_size=num_series)
    model = torch.load(model_dir + 'vae_gan_trained_model.pth').to(device)
    gen_data = None
    for noise_fake_batch in noise_fake:
        model.eval()
        with torch.no_grad():
            _, _, x_fake = model.decoder.forward(x=torch.concat(
                (noise_fake_batch['feature'], diff_data_batch_loader.dataset.first_day), dim=-1).to(device))
            gen_data = torch.relu(
                x_fake + diff_data_batch_loader.dataset.first_day.to(device)).cpu().detach().numpy()

    real_data = diff_data_batch_loader.dataset.feature.cpu().detach().numpy()
    np.save(log_dir + '/generated_data.npy', gen_data)
    np.save(log_dir + '/real_data.npy', real_data + first_day)


def main():
    args = arg_parse()
    # dir = "Cluster_4"
    # parent = "./experiments/states_covid_long_term"

    # experimental results directory
    # ========
    dir = args.experiment_name
    parent = args.parent_dir
    # ========
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
    # dir = "vae_gan_diff_shuffled_" + str(args.data_name) + "_batch_size_" + str(
    #     args.batch_size) + "_lr_" + str(
    #     args.lr) + "_decay_rate_" + str(args.opt_decay_rate) + "_epoch_" + str(args.epochs) + "_noise_dim_" + str(
    #     args.noise_dim) + "_hidden_dim_" + str(
    #     args.hidden_dim) + "_decoder_layer_" + str(args.decoder_layer) + "_encoder_layer_" + str(
    #     args.encoder_layer)
    # parent = "./experiments/smooth_covid_diff_qaunt"

    train_model(args, path)
    # path = "./experiments/vae_dann_seq_norm_LSTM_moment_len_30_cluster_3_clean_batch_size_50_lr_0.004_decay_rate_0.8_epoch_300_noise_dim_20_hidden_dim_30_decoder_layer_4_encoder_layer_2_d_layer_2_repeat_1"
    test(path + '/saved_model/', path, args)
    # recover_fake_diff(data_metric_dir=path + '/', args=args)
    # visualization(fake=fake_total_change, real=real_total_change, option='tsne', dir=path + '/')


if __name__ == '__main__':
    main()
