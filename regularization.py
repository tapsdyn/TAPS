import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class spar_loss(nn.Module):
    def __init__(self):
        super(spar_loss, self).__init__()

    def forward(self, flops_real, flops_ori, batch_size, den_target, lbda, L1_loss=False):

        # total sparsity
        flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        frames_per_clip = flops_tensor.shape[1]
        # block flops
        flops_conv = flops_tensor.mean(dim=0).sum()  # average flops of all resnet blocks per clip
        flops_ori = flops_ori.sum() + flops_conv1.mean() * frames_per_clip + flops_fc.mean()  # original model flops per clip
        flops_real = flops_conv + flops_conv1.mean() * frames_per_clip + flops_fc.mean()  # measured flops per clip
        # loss
        if L1_loss:
            rloss = lbda * torch.max(torch.zeros([1], device=flops_real.device), (flops_real / flops_ori - den_target))
        else:
            rloss = lbda * (flops_real / flops_ori - den_target) ** 2
        return rloss


class BatchShapingFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_m, beta_pdf_lut, beta_cdf_lut, beta_cdf_res, n):
        x_sort, idx_sort = torch.sort(x_m, 0)
        x_sort = torch.clamp(x_sort, min=1/beta_cdf_res, max=1-1/beta_cdf_res)
        x_sort_ints = torch.round(x_sort * beta_cdf_res) - 1
        x_sort_ints = x_sort_ints.to(torch.long)
        p_pdf = torch.squeeze(beta_pdf_lut(x_sort_ints))
        p_cdf = torch.squeeze(beta_cdf_lut(x_sort_ints))
        e_cdf = torch.unsqueeze(torch.arange(1, n+1, device=x_m.device) / (n + 1), 1)
        ctx.save_for_backward(idx_sort, p_pdf, p_cdf, e_cdf)
        loss = torch.mean(torch.sum(torch.square(e_cdf - p_cdf), dim=0))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        idx_sort, p_pdf, p_cdf, e_cdf = ctx.saved_tensors
        grad_sorted = -2.0 * p_pdf * (e_cdf - p_cdf)
        grad = grad_sorted.gather(0, idx_sort.argsort(0))
        return grad_output * grad, None, None, None, None, None


class BatchShapingLoss(nn.Module):
    def __init__(self, pdf_alpha, pdf_beta, beta_cdf_res, target_device):
        super(BatchShapingLoss, self).__init__()

        from scipy.stats import beta as scipy_beta
        import numpy as np

        beta_xs_in_np = np.linspace(1 / beta_cdf_res, 1 - 1 / beta_cdf_res, beta_cdf_res - 1)
        beta_pdf_out_np = scipy_beta.pdf(beta_xs_in_np, pdf_alpha, pdf_beta)
        beta_cdf_out_np = scipy_beta.cdf(beta_xs_in_np, pdf_alpha, pdf_beta)

        beta_pdf_lut_tensor = torch.FloatTensor(np.expand_dims(beta_pdf_out_np, 1)).to(target_device)
        self.beta_pdf_lut = nn.Embedding.from_pretrained(beta_pdf_lut_tensor, freeze=True)
        beta_cdf_lut_tensor = torch.FloatTensor(np.expand_dims(beta_cdf_out_np, 1)).to(target_device)
        self.beta_cdf_lut = nn.Embedding.from_pretrained(beta_cdf_lut_tensor, freeze=True)
        self.beta_cdf_res = beta_cdf_res
        self.batch_shaping_loss = BatchShapingFunc.apply

    def forward(self, x_m, gamma):
        loss = gamma * self.batch_shaping_loss(x_m, self.beta_pdf_lut, self.beta_cdf_lut, self.beta_cdf_res,
                                               x_m.shape[0])
        return loss


class Loss(nn.Module):
    def __init__(self, bs_pdf_alpha, bs_pdf_beta, bs_beta_cdf_res, target_device):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss()
        self.spar_loss = spar_loss()
        self.bs_loss = BatchShapingLoss(bs_pdf_alpha, bs_pdf_beta, bs_beta_cdf_res, target_device)

    def forward(self, output, targets, flops_real, flops_ori, batch_size, den_target, lbda, channel_gates_outputs,
                epoch=None, args=None):
        # output: [batch_size, num_classes] - model prediction per clip
        # targets: [batch_size, ] - ground truth class label per clip
        # flops_real - list:
        # flops_real[0]: [BatchSize, FramesPerClip, #ResNetLayers] - The actual measured flops per frame per residual block
        # flops_real[1]: scalar - flops (per 1 frame) of the first convolution in the network
        # flops_real[2]: scalar - flops (per clip) of the FC layer
        # flops_ori: [1, #ResNetLayers] - flops (per clip) of the static model ops (conv1+conv2+downsampling+channels policy) at each ResNetLayer
        # batch_size: - scalar - number of clips in batch
        # channel_gates_outputs: List with length #ResNetLayers
        #           For each ResNet block K, channel_gates_outputs[K] has shape [BatchSize, NumSegments, NumChannels, 1]
        #           the values are the gate "soft mask" / probability of being open (1) or closed (0)
        #           values are in the range 0-1 (after sigmoid)
        closs = self.task_loss(output, targets)
        if args is None:
            sloss = self.spar_loss(flops_real, flops_ori, batch_size, den_target, lbda)
            bs_loss = torch.tensor(0.0, device=closs.device)
        else:
            closs = self.task_loss(output, targets)
            if epoch < args.eff_loss_after:
                lbda = 0
            elif 0 <= args.sparsity_warmup_start < args.sparsity_warmup_end:
                if epoch <= args.sparsity_warmup_start:
                    sparsity_loss_factor = 0.0
                elif args.sparsity_warmup_start < epoch < args.sparsity_warmup_end:
                    sparsity_loss_factor = (epoch - args.sparsity_warmup_start) / \
                                           (args.sparsity_warmup_end - args.sparsity_warmup_start)
                else:
                    sparsity_loss_factor = 1.0
                lbda = lbda * sparsity_loss_factor

            sloss = self.spar_loss(flops_real, flops_ori, batch_size, den_target, lbda, args.spars_loss_L1)

            if args.bs_gamma > 0:
                # batch shaping loss
                bs_gamma = args.bs_gamma
                if epoch < args.bs_loss_start_epoch:
                    bs_gamma = 0.0
                elif args.bs_loss_stop_epoch > 0:
                    bs_gamma = ((args.bs_loss_stop_epoch - epoch) /
                                (args.bs_loss_stop_epoch - args.bs_loss_start_epoch)) * bs_gamma
                if bs_gamma > 0:
                    # channel_gates_outputs[K]: [batch_s, n_segments, n_channels, 1] -> [batch_s*n_segments, n_channels]
                    if args.bs_from_block > 0:
                        channel_gates_outputs = channel_gates_outputs[args.bs_from_block:]
                    def flatten_gate_outpus(input):
                        return torch.squeeze(input.view((-1,) + input.shape[2:]))
                    blocks_x_m = list(map(flatten_gate_outpus, channel_gates_outputs))
                    # x_m (gate outputs): [batch_size * num_segments, total_num_gates]
                    x_m = torch.hstack(blocks_x_m)
                    bs_loss = self.bs_loss(x_m, bs_gamma)
                else:
                    bs_loss = torch.tensor(0.0, device=closs.device)
            else:
                bs_loss = torch.tensor(0.0, device=closs.device)

        return closs, sloss, bs_loss
