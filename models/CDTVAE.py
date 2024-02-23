import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import math

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

def KL_divergence(mu_1, logvar_1, mu_2, log_var_2):
    # log(var2) - log(var1) + (var1 + (mu1 - mu2)**2) / var2 + 1 /2
    return torch.mean(torch.sum(-0.5 * logvar_1 + torch.div((torch.exp(logvar_1) + (mu_1 - mu_2) ** 2), 2) - 0.5, dim=[-1, -2]), dim=0)

class Prior(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, args):
        super(Prior, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim
        self.num_layers = args.encoder_layer

        # Encoder for encode the original x_0 to z
        self.mean = nn.ModuleList()
        self.mean.append(
            nn.Linear(input_dim, hidden_dim))
        for _ in range(args.encoder_layer):
            self.mean.append(nn.Linear(hidden_dim, hidden_dim))
        self.mean.append(nn.Linear(hidden_dim, out_dim))

        self.logvar = nn.ModuleList()
        self.logvar.append(
            nn.Linear(input_dim, hidden_dim))
        for _ in range(args.encoder_layer):
            self.logvar.append(nn.Linear(hidden_dim, hidden_dim))
        self.logvar.append(nn.Linear(hidden_dim, out_dim))
    
    @staticmethod
    def reparametrize(mu, logvar):
        std = 1  # e**(x/2)
        eps = torch.FloatTensor(mu.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.add_(mu)  # reparametrized z~N(mu, var)
    
    def prior_forward(self, x):
        mu = self.mean[0](x)
        for i in range(self.num_layers):
            mu = self.mean[i+1](mu)
            mu = torch.nn.LeakyReLU()(mu)
        mu = self.mean[-1](mu)
        logvar = self.logvar[0](x)
        for i in range(self.num_layers):
            logvar = self.logvar[i+1](logvar)
            logvar = torch.nn.LeakyReLU()(logvar)
        logvar = self.logvar[-1](logvar)
        z = self.reparametrize(mu.clone(), 0)
        return z, mu, logvar

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, args):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim

        # Encoder for encode the original x to z
        # Encoder for encode the original x to z
        self.mean = nn.ModuleList()
        self.mean.append(
            nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1,
                    batch_first=True))
        self.mean.append(nn.Linear(input_dim, hidden_dim))
        self.mean.append(nn.LSTM(input_size=hidden_dim, hidden_size=out_dim, num_layers=args.encoder_layer,
                                 batch_first=True))  # mean

        self.logvar = nn.ModuleList()
        self.logvar.append(
            nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1,
                    batch_first=True))
        self.logvar.append(nn.Linear(input_dim, hidden_dim))
        self.logvar.append(nn.LSTM(input_size=hidden_dim, hidden_size=out_dim, num_layers=args.encoder_layer,
                                   batch_first=True))  # log(variance), a general trick

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()  # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mu)  # reparametrized z~N(mu, var)

    def encoder_forward(self, x, args):
        x_mean, _ = self.mean[0](x)  # RNN's activation function is embedded in PyTorch
        x_mean = self.mean[1](x_mean)
        mu, _ = self.mean[2](x_mean)

        x_logvar, _ = self.logvar[0](x)
        x_logvar = self.logvar[1](x_logvar)
        logvar, _ = self.logvar[2](x_logvar)

        z = self.reparametrize(mu.clone(), logvar.clone())
        # calculation of KL divergence (regulation loss)
        # KL_div = torch.mean(0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[-1, -2]), dim=0)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(Decoder, self).__init__()
        self.mean_decoder = nn.ModuleList()
        self.mean_decoder.append(
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=args.decoder_layer))
        self.mean_decoder.append(nn.Linear(hidden_dim, output_dim))

        self.logvar_decoder = nn.ModuleList()
        self.logvar_decoder.append(
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=args.decoder_layer))
        self.logvar_decoder.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        mean_x, _ = self.mean_decoder[0](x)
        mu_decoder = self.mean_decoder[1](mean_x)

        log_var_x, _ = self.logvar_decoder[0](x)
        logvar_decoder = self.logvar_decoder[1](log_var_x)

        repemetrimized_x = self.reparametrize(mu=mu_decoder.clone(), logvar=logvar_decoder.clone())
        return mu_decoder, logvar_decoder, repemetrimized_x

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()  # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mu)  # reparametrized z~N(mu, var)

    @staticmethod
    def reconstruction_loss(mu, logvar, real):
        std = logvar.mul(0.5).exp()
        temp = 0.5 * torch.sum((torch.div((mu - real.clone()), std) ** 2), dim=[-1, -2]) + 0.5 * torch.sum(logvar,
                                                                                                           dim=[-1, -2])
        reconstruct_loss = torch.mean(temp)
        return reconstruct_loss

class CDTVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, output_dim, args):
        super(CDTVAE, self).__init__()
        self.encoder = Encoder(input_dim=input_dim + input_dim, hidden_dim=hidden_dim, out_dim=noise_dim, args=args)
        self.decoder = Decoder(input_dim=noise_dim + input_dim, hidden_dim=hidden_dim, output_dim=output_dim, args=args)
        self.prior = Prior(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=noise_dim, args=args)

    def forward(self, x, real, first_day, args):
        z, mu_z, log_var_z = self.encoder.encoder_forward(x=x, args=args)
        prior_z, prior_mu, prior_logvar = self.prior.prior_forward(x=first_day)
        # print('the shape of z and prior_z')
        # print(z.shape, prior_z.shape)
        kl_div = KL_divergence(mu_1=mu_z.clone(), logvar_1=log_var_z.clone(), mu_2=prior_mu.clone(), log_var_2=prior_logvar.clone())
        mu_decoder, logvar_decoder, repemetrimized_x = self.decoder.forward(x=torch.concat((z, first_day), dim=-1))
        reconstruction_loss = self.decoder.reconstruction_loss(mu=mu_decoder.clone(), logvar=logvar_decoder.clone(), real=real.clone())
        return kl_div, repemetrimized_x, reconstruction_loss
