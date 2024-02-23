import torch
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard.summary import hparams


def construct_seq_mask(row_seq_len, col_seq_len):
    mask = (torch.triu(torch.ones(row_seq_len, col_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    mask_np = mask.cpu().numpy()
    return mask


def construct_src_tgt_mask(row_seq_len, col_seq_len):
    mask = (torch.tril(torch.ones(row_seq_len, col_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    mask_np = mask.cpu().numpy()
    return mask


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900],
                                                   gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


# class Writer(SummaryWriter):
#     def add_hparams(
#         self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
#     ):
#         torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
#         if type(hparam_dict) is not dict or type(metric_dict) is not dict:
#             raise TypeError('hparam_dict and metric_dict should be dictionary.')
#         exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
#
#         logdir = self._get_file_writer().get_logdir()
#         with SummaryWriter(log_dir=logdir) as w_hp:
#             w_hp.file_writer.add_summary(exp)
#             w_hp.file_writer.add_summary(ssi)
#             w_hp.file_writer.add_summary(sei)
#             for k, v in metric_dict.items():
#                 w_hp.add_scalar(k, v)
if __name__ == '__main__':
    temp = construct_seq_mask(1,1)
    temp_zero = torch.zeros((5,3)).unsqueeze(1)
    temp_one = torch.ones((5,3)).unsqueeze(1)
    temp_cat = torch.cat((temp_zero, temp_one), dim=1)
    temp_cat_1 = temp_cat + 1
    print()

