import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import math

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


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
        KL_div = torch.mean(0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[-1, -2]), dim=0)
        return KL_div, z


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

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, args, seq_len=30):
        super(Discriminator, self).__init__()
        self.dis = nn.ModuleList()
        self.dis.append(nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=args.d_layer, batch_first=True))
        self.dis.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, alpha, epoch):
        y = GradReverse.apply(x, alpha).clone()
        y, _ = self.dis[0](y)
        y = self.dis[1](y)
        dis_x = torch.sigmoid(y)
        return dis_x

    @staticmethod
    def loss(pred, label):
        loss = nn.BCELoss(reduction='sum')
        bce_loss = loss(pred, label) / pred.shape[0]
        return bce_loss

class VAE_GAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, output_dim, args):
        super(VAE_GAN, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=noise_dim, args=args)
        self.decoder = Decoder(input_dim=noise_dim, hidden_dim=hidden_dim, output_dim=output_dim, args=args)
        self.discriminator = Discriminator(input_dim=output_dim, hidden_dim=hidden_dim, out_dim=1, args=args)


    def forward(self, x, real, fake, alpha, epoch, args):
        kl_div, z = self.encoder.encoder_forward(x=x, args=args)
        mu_decoder, logvar_decoder, repemetrimized_x = self.decoder.forward(x=z)
        _,_,noise_reconstruct = self.decoder.forward(x=fake)
        reconstruction_loss = self.decoder.reconstruction_loss(mu=mu_decoder.clone(), logvar=logvar_decoder.clone(), real=real.clone())
        
        fake_reconstrcuted_prob = self.discriminator.forward(x=repemetrimized_x, alpha=alpha, epoch=epoch)
        fake_prob = self.discriminator.forward(x=noise_reconstruct, alpha=alpha, epoch=epoch)
        real_prob = self.discriminator.forward(x=real, alpha=alpha, epoch=epoch)

        bce_loss = self.discriminator.loss(pred=real_prob, label=torch.zeros(real_prob.shape).to(real_prob.device))
        bce_loss = bce_loss + self.discriminator.loss(pred=fake_prob, label=torch.ones(real_prob.shape).to(real_prob.device))
        bce_loss = bce_loss + self.discriminator.loss(pred=fake_reconstrcuted_prob,label=torch.ones(real_prob.shape).to(real_prob.device))

        return kl_div, repemetrimized_x, reconstruction_loss, bce_loss
