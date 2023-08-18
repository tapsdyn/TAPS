import numpy as np
import math
import torch
import torch.nn.functional as F
from ops.net_flops_table import get_gflops_params

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_multi_hot(test_y, classes, assumes_starts_zero=True):
    bs = test_y.shape[0]
    label_cnt = 0

    if not assumes_starts_zero:
        for label_val in torch.unique(test_y):
            if label_val >= 0:
                test_y[test_y == label_val] = label_cnt
                label_cnt += 1

    gt = torch.zeros(bs, classes + 1)
    for i in range(test_y.shape[1]):
        gt[torch.LongTensor(range(bs)), test_y[:, i]] = 1

    return gt[:, :classes]

def cal_map(output, old_test_y):
    batch_size = output.size(0)
    num_classes = output.size(1)
    ap = torch.zeros(num_classes)
    test_y = old_test_y.clone()

    gt = get_multi_hot(test_y, num_classes, False)

    probs = F.softmax(output, dim=1)

    rg = torch.range(1, batch_size).float()
    for k in range(num_classes):
        scores = probs[:, k]
        targets = gt[:, k]
        _, sortind = torch.sort(scores, 0, True)
        truth = targets[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)
        ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
    return ap.mean()*100, ap*100


class Recorder:
    def __init__(self, larger_is_better=True):
        self.history=[]
        self.larger_is_better=larger_is_better
        self.best_at=None
        self.best_val=None

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x>y
        else:
            return x<y

    def update(self, val):
        self.history.append(val)
        if len(self.history)==1 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = len(self.history)-1

    def is_current_best(self):
        return self.best_at == len(self.history)-1

    def at(self, idx):
        return self.history[idx]

def adjust_learning_rate(optimizer, epoch, length, iteration, lr_type, lr_steps, args):
    if lr_type == 'step':
        decay = (1 / args.decay_step) ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    elif lr_type == 'linear':
        factor = min(1.0, (epoch * length + iteration + 1)/(args.warmup_epochs * length))
        lr = args.lr * factor
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        if lr_type != 'linear':
            param_group['weight_decay'] = decay * param_group['decay_mult']


def count_conv2d_flops(input_data_shape, conv):
    n, c_in, h_in, w_in = input_data_shape
    h_out = (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w_out = (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    c_out = conv.out_channels
    bias = 1 if conv.bias is not None else 0
    flops = n * c_out * h_out * w_out * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
    return flops, (n, c_out, h_out, w_out)


def product(tuple1):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tuple1:
        prod = prod * x
    return prod

def count_bn_flops(input_data_shape):
    flops = product(input_data_shape) * 2
    output_data_shape = input_data_shape
    return flops, output_data_shape


def count_relu_flops(input_data_shape):
    flops = product(input_data_shape) * 1
    output_data_shape = input_data_shape
    return flops, output_data_shape


def count_fc_flops(input_data_shape, fc):
    output_data_shape = input_data_shape[:-1] + (fc.out_features, )
    flops = product(output_data_shape) * (fc.in_features + 1)
    return flops, output_data_shape

def init_gflops_table(model, args):
    if "cgnet" in args.arch:
        base_model_gflops = 1.8188 if "net18" in args.arch else 4.28
        params = get_gflops_params(args.arch, args.reso_list[0], args.num_class, -1, args=args)[1]
    else:
        base_model_gflops, params = get_gflops_params(args.arch, args.reso_list[0], args.num_class, -1, args=args)

    gflops_list = model.base_model.count_flops((1, 1, 3, args.reso_list[0], args.reso_list[0]))
    if "AdaBNInc" in args.arch:
        gflops_list, g_meta = gflops_list
    else:
        g_meta = None
    print("Network@%d (%.4f GFLOPS, %.4f M params) has %d blocks" % (args.reso_list[0], base_model_gflops, params, len(gflops_list)))
    for i, block in enumerate(gflops_list):
        print("block", i, ",".join(["%.4f GFLOPS" % (x / 1e9) for x in block]))
    return base_model_gflops, gflops_list, g_meta


def compute_gflops_by_mask(mask_tensor_list, base_model_gflops, gflops_list, g_meta, args):
    upperbound_gflops = base_model_gflops
    real_gflops = base_model_gflops

    # TODO YD: dynstc is specified here to create reference until replaced by "online" flops calculation
    if "bate" in args.arch or "dynstc" in args.arch:
        for m_i, mask in enumerate(mask_tensor_list):
            #compute precise GFLOPS
            upsave = torch.zeros_like(mask[:, :, :, 0]) # B*T*C*K->B*T*C
            for t in range(mask.shape[1]-1):
                if args.gate_history:
                    upsave[:, t] = (1 - mask[:, t, :, -1]) * (1 - mask[:, t + 1, :, -2])
                else:
                    upsave[:, t] = 1 - mask[:, t, :, -1] # since no reusing, as long as not keeping, save from upstream conv
            upsave[:, -1] = 1 - mask[:, -1, :, -1]  # original (buggy) code: 1 - mask[:, t, :, -1]
            upsave = torch.mean(upsave)

            if args.gate_no_skipping: # downstream conv gflops' saving is from skippings
                downsave = upsave * 0
            else:
                downsave = torch.mean(mask[:, :, :, 0])

            conv_offset = 0
            real_count = 1.

            layer_i = m_i
            up_flops = gflops_list[layer_i][0 + conv_offset] / 1e9
            down_flops = gflops_list[layer_i][1 + conv_offset] * real_count / 1e9
            embed_conv_flops = gflops_list[layer_i][-1] * real_count / 1e9

            upperbound_gflops = upperbound_gflops - downsave * (down_flops - embed_conv_flops) # in worst case, we only compute saving from downstream conv
            real_gflops = real_gflops - upsave * up_flops - downsave * (down_flops - embed_conv_flops)
    elif "AdaBNInc" in args.arch:
        for m_i, mask in enumerate(mask_tensor_list):
            upsave = torch.zeros_like(mask[:, :, :, 0])  # B*T*C*K->B*T*C
            for t in range(mask.shape[1]-1):
                if args.gate_history:
                    upsave[:, t] = (1 - mask[:, t, :, -1]) * (1 - mask[:, t + 1, :, -2])
                else:
                    upsave[:, t] = 1 - mask[:, t, :, -1] # since no reusing, as long as not keeping, save from upstream conv
            upsave[:, -1] = 1 - mask[:, t, :, -1]
            upsave = torch.mean(upsave, dim=[0,1]) # -> C

            if len(gflops_list[m_i]) == 7:
                _a,_b,_c,_d = g_meta[m_i]
                upsaves = [torch.mean(upsave[:_a]),
                           torch.mean(upsave[_a:_a + _b]),
                           torch.mean(upsave[_a + _b:_a + _b + _c]),
                           torch.mean(upsave[_a + _b + _c:])]
                out_corr_list = [0, 2, 5, 6]  # to the id of last convs in each partition
                if m_i < len(gflops_list)-1 and len(gflops_list[m_i+1]) == 5:
                    next_in_corr_list = [0, 2]
                else:
                    next_in_corr_list = [0, 1, 3, 6]
            elif len(gflops_list[m_i]) == 5:
                _a, _b, _c= g_meta[m_i]
                upsaves = [torch.mean(upsave[:_a]),
                            torch.mean(upsave[_a:_a+_b]),
                            torch.mean(upsave[_a+_b:])]
                out_corr_list = [1, 4]  # to the id of last convs in each partition
                next_in_corr_list = [0, 1, 3, 6]
            up_flops_save = sum([upsaves[f_i] * gflops_list[m_i][out_corr_list[f_i]] for f_i in range(len(out_corr_list))]) / 1e9
            if args.gate_no_skipping: # downstream conv gflops' saving is from skippings
                downsave = upsaves[0] * 0
            else:
                downsave = torch.mean(mask[:, :, :, 0])
            down_flops_save = up_flops_save * 0
            if m_i < len(mask_tensor_list)-1:
                # to the id of first convs in each partition in the next layer
                down_flops_save = downsave * sum([gflops_list[m_i+1][next_in_corr_list[f_i]] for f_i in range(len(next_in_corr_list))]) / 1e9
            upperbound_gflops = upperbound_gflops - down_flops_save
            real_gflops = real_gflops - up_flops_save - down_flops_save
    else:
        # s0 for sparsity savings
        # s1 for history
        s0 = [1 - 1.0 * torch.sum(mask[:, :, 1]) / torch.sum(mask[:, :, 0]) for mask in mask_tensor_list]

        savings = sum([s0[i] * gflops_list[i][0] * (1 - 1.0 / args.partitions) for i in range(len(gflops_list))])
        real_gflops = base_model_gflops - savings / 1e9
        upperbound_gflops = real_gflops

    return upperbound_gflops, real_gflops


def compute_gflops_dynstc(dyn_outputs, num_segments, batch_size, thop_base_model_gflops):

    frames_dynamic_resnet_layers_flops = dyn_outputs['flops_real'][0]  # [BatchSize, FramesPerClip, #ResNetLayers] (w.o. policy)
    first_conv_flops = dyn_outputs['flops_real'][1][0]  # [scalar]: flops (per 1 frame) of the first conv
    static_resnet_layers_flops = dyn_outputs['flops_ori'][0] # [1, #ResNetLayers]: flops (per clip) of the STATIC model ops (conv1+conv2+downsampling+channels policy) at each ResNetLayer
    upperbound_resnet_layer_flops = torch.sum(dyn_outputs['flops_upperbound'], 0)  # [1, #ResNetLayers]: flops (per clip) at each ResNetLayer, considering only downstrram (2nd conv in ResNet block) flops saving
    downsample_resnet_layer_flops = dyn_outputs['flops_downsample'][0]  # [1, #ResNetLayers]: flops (per clip) of the downsampling (if exists) at each ResNetLayer
    channels_mask_flops = dyn_outputs['flops_channels_mask'][0]  # [1, #ResNetLayers]: flops (per clip) of the channels policy net at each ResNetLayer

    base_model_gflops = (torch.sum(static_resnet_layers_flops / num_segments) + first_conv_flops) / 10 ** 9
    dynamic_model_const_flops = first_conv_flops + (torch.sum(downsample_resnet_layer_flops) +
                                                    torch.sum(channels_mask_flops)) / num_segments
    real_flops = (torch.sum(frames_dynamic_resnet_layers_flops) / (batch_size * num_segments) +
                     dynamic_model_const_flops) / 10 ** 9
    upperbound_flops = (torch.sum(upperbound_resnet_layer_flops) / (batch_size * num_segments) +
                           dynamic_model_const_flops) / 10 ** 9
    # TODO YD: remaining (constant) difference is due to static model flops diff to thop measurements
    const_static_diff = thop_base_model_gflops - base_model_gflops
    real_flops_fixed = real_flops + const_static_diff
    upb_flops_fixed = upperbound_flops + const_static_diff
    return upb_flops_fixed, real_flops_fixed


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class ExpAnnealing(object):
    r"""
    Args:
        T_max (int): Maximum number of iterations.
        eta_ini (float): Initial density. Default: 1.
        eta_min (float): Minimum density. Default: 0.
    """

    def __init__(self, T_ini, eta_ini=1, eta_final=0, up=False, alpha=1):
        self.T_ini = T_ini
        self.eta_final = eta_final
        self.eta_ini = eta_ini
        self.up = up
        self.last_epoch = 0
        self.alpha = alpha

    def get_lr(self, epoch):
        if epoch < self.T_ini:
            return self.eta_ini
        elif self.up:
            return self.eta_ini + (self.eta_final-self.eta_ini) * (1-
                   math.exp(-self.alpha*(epoch-self.T_ini)))
        else:
            return self.eta_final + (self.eta_ini-self.eta_final) * math.exp(
                   -self.alpha*(epoch-self.T_ini))

    def step(self):
        self.last_epoch += 1
        return self.get_lr(self.last_epoch)

