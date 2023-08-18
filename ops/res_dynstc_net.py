import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

from ops.utils import count_conv2d_flops
from ops.utils import conv2d_out_dim
import torchvision
from ops.policy_block import PolicyBlock, ATTENTION_PE_IN_CHANNELS, handcraft_policy_for_masks


__all__ = ['ResDynSTCNet', 'res18_dynstc_net', 'res50_dynstc_net']

Skip = 0
Reuse = 1
Keep = 2

model_urls = {
    'res18_dynstc_net': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'res50_dynstc_net': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def list_sum(obj):
    if isinstance(obj, list):
        if len(obj)==0:
            return 0
        else:
            return sum(list_sum(x) for x in obj)
    else:
        return obj


def shift(x, n_segment, fold_div=3, inplace=False, online_shift=False, shift_copy_pad=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    if inplace:
        # Due to some out of order error when performing parallel computing.
        # May need to write a CUDA kernel.
        raise NotImplementedError
        # out = InplaceShift.apply(x, fold)
    else:
        if shift_copy_pad:
            out = x.clone()
        else:
            out = torch.zeros_like(x)

        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        if not online_shift:
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        else:  # shift online
            out[:, :, :fold] = x[:, :, :fold]  # not shift
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)


def get_hmap(out, args, **kwargs):
    out_reshaped = out.view((-1, args.num_segments) + out.shape[1:])

    if args.gate_history:
        h_map_reshaped = torch.zeros_like(out_reshaped)
        h_map_reshaped[:, 1:] = out_reshaped[:, :-1]
    else:
        return None

    if args.gate_history_detach:
        h_map_reshaped = h_map_reshaped.detach()

    h_map_updated = h_map_reshaped.view((-1,) + out_reshaped.shape[2:])
    return h_map_updated


def fuse_out_with_mask(out, mask, h_map, args, soft_mask=None):

    if mask is not None:
        if soft_mask is not None:
            mask = mask * soft_mask
        if args.pn_num_outputs == 3:
            # Skip, Reuse, Keep
            out = out * mask[:, :, Keep].unsqueeze(-1).unsqueeze(-1)
        elif args.pn_num_outputs == 2 or args.pn_num_outputs == 1:
            # Skip, Keep
            # Now Keep isn't 2
            out = out * mask[:, :, -1].unsqueeze(-1).unsqueeze(-1)
        if args.gate_history:
            # Skip, Reuse, Keep
            out = out + h_map * mask[:, :, Reuse].unsqueeze(-1).unsqueeze(-1)

    return out


def count_dyn_conv2d_flops(input_data_shape, conv, channels_mask, upstream_conv):
    n, c_in, h_in, w_in = input_data_shape
    h_out = (h_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w_out = (w_in + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    c_out = conv.out_channels
    bias = 1 if conv.bias is not None else 0

    # compute precise GFLOP
    ofm_size = h_out * w_out
    frames_per_clip = channels_mask.shape[1]
    out_active_pixels = ofm_size * torch.ones((n // frames_per_clip, frames_per_clip),
                                              device=channels_mask.device)  # [batch_size, frames_per_clip]
    frames_per_clip = channels_mask.shape[1]
    pn_num_outputs = channels_mask.shape[-1] # self.args.pn_num_outputs
    gate_history = channels_mask.shape[-1] > 2

    if upstream_conv:
        # batch_size*frames*channels*K->batch_size*frames*channels
        out_channel_off = torch.zeros_like(channels_mask[:, :, :, 0], device=channels_mask.device)

        for t in range(frames_per_clip - 1):
            if gate_history:
                # Skip, Reuse, Keep
                out_channel_off[:, t, :] = (1 - channels_mask[:, t, :, Keep]) * (1 - channels_mask[:, t + 1, :, Reuse])
            else:
                out_channel_off[:, t, :] = 1 - channels_mask[:, t, :, -1]  # since no reusing, as long as not keeping, save from upstream conv

        if pn_num_outputs == 3:
            # Skip, Reuse, Keep
            out_channel_off[:, -1, :] = 1 - channels_mask[:, -1, :, Keep]
        elif pn_num_outputs == 2 or pn_num_outputs == 1:
            # Skip, Keep
            # Now Keep isn't 2
            out_channel_off[:, -1, :] = 1 - channels_mask[:, -1, :, -1]
        out_active_channels = c_out - torch.sum(out_channel_off, dim=2)  # [batch_size, frames_per_clip]

        flops_per_frame = out_active_channels * out_active_pixels * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
        flops_upper_bound_mat = c_out * out_active_pixels * (c_in // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
    else:
        # downstream conv flops saving is from skippings
        if pn_num_outputs == 1:
            in_channel_off = 1.0 - channels_mask[:, :, :, 0]  # [batch_size, frames_per_clip, channels]
        else:
            in_channel_off = channels_mask[:, :, :, 0]  # [batch_size, frames_per_clip, channels]


        in_active_channels = torch.add(torch.neg(torch.sum(in_channel_off, dim=2)),
                                       c_in)  # [batch_size, frames_per_clip]

        flops_per_frame = c_out * out_active_pixels * (in_active_channels // conv.groups * conv.kernel_size[0] * conv.kernel_size[1] + bias)
        flops_upper_bound_mat = flops_per_frame

    flops = flops_per_frame.reshape((-1,))
    flops_upper_bound = torch.sum(flops_upper_bound_mat, dim=1) # [batch_size,]

    return flops, flops_upper_bound, (n, c_out, h_out, w_out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, h, w, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, shared=None, shall_enable=None, args=None):
        super(BasicBlock, self).__init__()
        self.args = args
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # conv 1
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.FBS = nn.Linear(in_features=inplanes,
                              out_features=planes,
                              bias=True)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # conv 2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # misc
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.shall_enable = shall_enable
        self.num_channels = planes
        self.adaptive_policy = not any([self.args.gate_all_zero_policy,
                                        self.args.gate_all_one_policy,
                                        self.args.gate_only_current_policy,
                                        self.args.gate_random_soft_policy,
                                        self.args.gate_random_hard_policy,
                                        self.args.gate_threshold])
        if self.shall_enable==False and self.adaptive_policy:
            self.adaptive_policy=False
            self.use_current=True
        else:
            self.use_current=False

        if self.adaptive_policy:
            self.policy_net = PolicyBlock(in_planes=inplanes, out_planes=planes, norm_layer=norm_layer, shared=shared, args=args)

        if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
            self.gate_hist_conv = conv3x3(planes, planes, groups=planes)
            if self.args.gate_history_conv_type == 'ghostbnrelu':
                self.gate_hist_bnrelu = nn.Sequential(norm_layer(planes), nn.ReLU(inplace=True))
        # flops
        input_data_shape = (args.num_segments, inplanes, h, w)
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        flops_conv1_full = torch.Tensor([conv1_flops])
        flops_conv2_full = torch.Tensor([conv2_flops])
        self.flops_downsample = torch.Tensor([downsample0_flops])
        if self.adaptive_policy:
            self.flops_channels_policy = torch.Tensor([self.policy_net.flops * args.num_segments])
        else:
            self.flops_channels_policy = torch.Tensor([0])
        self.flops_full = flops_conv1_full + flops_conv2_full + self.flops_downsample + self.flops_channels_policy

    def count_flops(self, input_data_shape, **kwargs):
        # TODO YD: This is kept as a ref for comparison with new code
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0
        return [conv1_flops, conv2_flops, downsample0_flops, 0], conv2_out_shape

    def forward(self, input, **kwargs):
        x, flops = input
        identity = x
        mask, soft_mask = None, None
        h_map_updated = None

        # shift operations
        if self.args.shift:
            x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False,
                      online_shift=self.args.online_shift, shift_copy_pad=self.args.shift_copy_pad)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # gate functions
        disable_hard_channels_mask = kwargs['disable_hard_channels_mask'] if 'disable_hard_channels_mask' in\
                                                                             kwargs else False
        disable_soft_channels_mask = kwargs['disable_soft_channels_mask'] if 'disable_soft_channels_mask' in\
                                                                             kwargs else False
        if self.args.disable_channelwise_masking:
            pass
        else:
            h_map_updated = get_hmap(out, self.args, **kwargs)
            if self.adaptive_policy:
                if disable_hard_channels_mask and disable_soft_channels_mask and\
                        not(self.args.bs_policy_in_attached):
                    x_policy_input = x.clone().detach()
                else:
                    x_policy_input = x
                mask, soft_mask = self.policy_net(x_policy_input, **kwargs)
                if disable_hard_channels_mask:
                    if self.args.hard_gates_disable_outer_mask:
                        outer_mask = torch.zeros_like(mask)
                        outer_mask[:, :, -1] = 1.
                        mask = outer_mask
                    else:
                        mask[mask!=0] = 0.
                        mask[:, :, -1] = 1.
            else:
                mask = handcraft_policy_for_masks(x, out, self.num_channels, self.use_current, self.args)

        soft_mask_in_fuse = None if disable_soft_channels_mask else soft_mask
        out = fuse_out_with_mask(out, mask, h_map_updated, self.args, soft_mask=soft_mask_in_fuse)
        if self.args.FBS_scaling_only:
            s = F.adaptive_avg_pool2d(torch.abs(x), (1, 1)).view(x.size()[0], -1)
            t = F.relu(self.FBS(s))
            t = t / torch.sum(t, dim=1).unsqueeze(1) * out.shape[-1]
            out = out * t.unsqueeze(2).unsqueeze(3)

        x2 = out
        out = self.conv2(out)
        out = self.bn2(out)

        # gate functions
        out = fuse_out_with_mask(out, mask=None, h_map=None, args=self.args)
        mask2 = None

        if self.downsample0 is not None:
            y = self.downsample0(identity)
            identity = self.downsample1(y)
        out += identity
        out = self.relu(out)

        # flops
        if self.args.disable_channelwise_masking:
            # In this case we have 3 unused options.
            #
            mask = torch.zeros(identity.shape[0], identity.shape[1], 3, device=identity.device)
            mask[:, :, Keep] = torch.ones_like(mask[:, :, Keep])

        flops_blk = self.get_flops(mask, x.shape)
        flops = torch.cat((flops, flops_blk.unsqueeze(0)))

        if soft_mask is not None:
            soft_mask = soft_mask.view((-1, self.args.num_segments) + mask.shape[1:])

        return out, mask.view((-1, self.args.num_segments) + mask.shape[1:]), mask2, flops, soft_mask


    def get_flops(self, mask_c, input_data_shape):
        # mask_c: [NumFrames, NumChannels, 3] -> 3: col0:skip,col1:reuse,col2:keep
        channels_masks = mask_c.view((-1, self.args.num_segments) + mask_c.shape[1:])  # [BatchesNum, FramesPerClip, NumChannels, 3]

        return self.get_flops_per_frame(channels_masks, input_data_shape)

    def get_flops_per_frame(self, channels_masks, input_data_shape):
        # conv1
        flops_conv1, flops_upper_bound_conv1, conv1_out_shape = count_dyn_conv2d_flops(input_data_shape, self.conv1,
                                                                                       channels_masks,
                                                                                       upstream_conv=True)
        # conv2
        flops_conv2, flops_upper_bound_conv2, conv2_out_shape = count_dyn_conv2d_flops(conv1_out_shape, self.conv2,
                                                                                       channels_masks,
                                                                                       upstream_conv=False)
        # total
        flops = flops_conv1 + flops_conv2
        flops_upper_bound = torch.unsqueeze(torch.sum(flops_upper_bound_conv1 + flops_upper_bound_conv2), 0)
        # flops: [1, batch_size x frames_per_clip]
        # flops upper bound: scalar
        return torch.cat((flops, flops_upper_bound, self.flops_downsample.to(flops.device),
                          self.flops_channels_policy.to(flops.device), self.flops_full.to(flops.device)))


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, h, w, stride=1, downsample0=None, downsample1=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, shared=None, shall_enable=None, args=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        # conv 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # conv 2
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # conv 3
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        # misc
        self.relu = nn.ReLU(inplace=True)
        self.downsample0 = downsample0
        self.downsample1 = downsample1
        self.stride = stride

        self.args = args
        self.shall_enable = shall_enable
        self.num_channels = width
        self.adaptive_policy = not any([self.args.gate_all_zero_policy,
                                        self.args.gate_all_one_policy,
                                        self.args.gate_only_current_policy,
                                        self.args.gate_random_soft_policy,
                                        self.args.gate_random_hard_policy,
                                        self.args.gate_threshold])
        if self.shall_enable==False and self.adaptive_policy:
            self.adaptive_policy=False
            self.use_current=True
        else:
            self.use_current=False

        if self.adaptive_policy:
            self.policy_net = PolicyBlock(in_planes=inplanes, out_planes=width, norm_layer=norm_layer, shared=shared, args=args)

        if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
            self.gate_hist_conv = conv3x3(width, width, groups=width)
            if self.args.gate_history_conv_type == 'ghostbnrelu':
                self.gate_hist_bnrelu = nn.Sequential(norm_layer(width), nn.ReLU(inplace=True))

        # flops
        input_data_shape = (args.num_segments, inplanes, h, w)
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        conv3_flops, conv3_out_shape = count_conv2d_flops(conv2_out_shape, self.conv3)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        flops_conv1_full = torch.Tensor([conv1_flops])
        flops_conv2_full = torch.Tensor([conv2_flops])
        flops_conv3_full = torch.Tensor([conv3_flops])
        self.flops_downsample = torch.Tensor([downsample0_flops])
        if self.adaptive_policy:
            self.flops_channels_policy = torch.Tensor([self.policy_net.flops * args.num_segments])
        else:
            self.flops_channels_policy = torch.Tensor([0])

        self.flops_full = flops_conv1_full + flops_conv2_full + flops_conv3_full + self.flops_downsample + \
                          self.flops_channels_policy

    def count_flops(self, input_data_shape, **kwargs):
        # TODO YD: This is kept as a ref for comparison with new code
        conv1_flops, conv1_out_shape = count_conv2d_flops(input_data_shape, self.conv1)
        conv2_flops, conv2_out_shape = count_conv2d_flops(conv1_out_shape, self.conv2)
        conv3_flops, conv3_out_shape = count_conv2d_flops(conv2_out_shape, self.conv3)
        if self.downsample0 is not None:
            downsample0_flops, _ = count_conv2d_flops(input_data_shape, self.downsample0)
        else:
            downsample0_flops = 0

        return [conv1_flops, conv2_flops, conv3_flops, downsample0_flops, 0], conv3_out_shape

    def forward(self, input, **kwargs):
        x, flops = input
        identity = x
        mask, soft_mask = None, None
        h_map_updated = None

        # shift operations
        if self.args.shift:
            x = shift(x, self.args.num_segments, fold_div=self.args.shift_div, inplace=False,
                      online_shift=self.args.online_shift, shift_copy_pad=self.args.shift_copy_pad)

        # conv 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # gate functions
        disable_hard_channels_mask = kwargs['disable_hard_channels_mask'] if 'disable_hard_channels_mask' in \
                                                                             kwargs else False
        disable_soft_channels_mask = kwargs['disable_soft_channels_mask'] if 'disable_soft_channels_mask' in \
                                                                             kwargs else False
        if self.args.disable_channelwise_masking:
            pass
        else:
            h_map_updated = get_hmap(out, self.args, **kwargs)
            if self.args.gate_history_conv_type in ['ghost', 'ghostbnrelu']:
                h_map_updated = self.gate_hist_conv(h_map_updated)
                if self.args.gate_history_conv_type == 'ghostbnrelu':
                    h_map_updated = self.gate_hist_bnrelu(h_map_updated)

            if self.adaptive_policy:
                if disable_hard_channels_mask and disable_soft_channels_mask and not(self.args.bs_policy_in_attached):
                    x_policy_input = x.clone().detach()
                else:
                    x_policy_input = x
                mask, soft_mask = self.policy_net(x_policy_input, **kwargs)
                if disable_hard_channels_mask:
                    if self.args.hard_gates_disable_outer_mask:
                        outer_mask = torch.zeros_like(mask)
                        outer_mask[:, :, -1] = 1.
                        mask = outer_mask
                    else:
                        mask[mask != 0] = 0.
                        mask[:, :, -1] = 1.
            else:
                mask = handcraft_policy_for_masks(x, out, self.num_channels, self.use_current, self.args)

        soft_mask_in_fuse = None if disable_soft_channels_mask else soft_mask
        out = fuse_out_with_mask(out, mask, h_map_updated, self.args, soft_mask=soft_mask_in_fuse)

        # conv 2
        x2 = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # gate functions
        conv2_output_channels_mask = None
        out = fuse_out_with_mask(out, mask=conv2_output_channels_mask, h_map=None, args=self.args)
        mask2 = None

        # conv 3
        out = self.conv3(out)
        out = self.bn3(out)

        out = fuse_out_with_mask(out, mask=None, h_map=None, args=self.args)

        # identity
        if self.downsample0 is not None:
            y = self.downsample0(identity)
            identity = self.downsample1(y)

        out += identity
        out = self.relu(out)

        # flops
        if self.args.disable_channelwise_masking:
            mask = torch.zeros(identity.shape[0], identity.shape[1], 3, device=identity.device)
            mask[:, :, Keep] = torch.ones_like(mask[:, :, Keep])
        flops_blk = self.get_flops(mask, x.shape, self.args.gate_history)
        flops = torch.cat((flops, flops_blk.unsqueeze(0)))

        if soft_mask is not None:
            soft_mask = soft_mask.view((-1, self.args.num_segments) + mask.shape[1:])

        return out, mask.view((-1, self.args.num_segments) + mask.shape[1:]), mask2, flops, soft_mask

    def get_flops(self, mask_c, input_data_shape, gate_history):
        # mask_c: [NumFrames, NumChannels, 3] -> 3: col0:skip,col1:reuse,col2:keep
        channels_masks = mask_c.view((-1, self.args.num_segments) + mask_c.shape[1:]) #[BatchesNum, FramesPerClip, NumChannels, 3]
        # conv1
        flops_conv1, flops_upper_bound_conv1, conv1_out_shape = count_dyn_conv2d_flops(input_data_shape, self.conv1,
                                                                                       channels_masks,
                                                                                       upstream_conv=True)
        # conv2
        flops_conv2, flops_upper_bound_conv2, conv2_out_shape = count_dyn_conv2d_flops(conv1_out_shape, self.conv2,
                                                                                       channels_masks,
                                                                                       upstream_conv=False)

        # conv3
        # use "compute all" dummy channels mask as masking of the 3rd convolution inputs is disabled
        policy_outputs = 3 if gate_history else 2
        conv3_channels_masks = torch.zeros(channels_masks.shape[0], channels_masks.shape[1],
                                           self.conv3.out_channels, policy_outputs, device=channels_masks.device)
        conv3_channels_masks[:, :, :, -1] = 1.
        flops_conv3, flops_upper_bound_conv3, conv3_out_shape = count_dyn_conv2d_flops(conv2_out_shape, self.conv3,
                                                                                       conv3_channels_masks,
                                                                                       upstream_conv=False)
        # total
        flops = flops_conv1 + flops_conv2 + flops_conv3
        flops_upper_bound = torch.unsqueeze(torch.sum(flops_upper_bound_conv1 + flops_upper_bound_conv2 +
                                                      flops_upper_bound_conv3), 0)
        # flops: [1, batch_size x frames_per_clip]
        # flops upper bound: scalar
        return torch.cat((flops, flops_upper_bound, self.flops_downsample.to(flops.device),
                          self.flops_channels_policy.to(flops.device), self.flops_full.to(flops.device)))


class ResDynSTCNet(nn.Module):

    def __init__(self, block, layers, h=224, w=224, num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, args=None):
        super(ResDynSTCNet, self).__init__()
        # block
        self.height, self.width = h, w
        # norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # conv1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.args = args

        self.gate_fc0s = None
        self.gate_fc1s = None

        self.relu = nn.ReLU(inplace=True)
        h = conv2d_out_dim(h, kernel_size=7, stride=2, padding=3)
        w = conv2d_out_dim(w, kernel_size=7, stride=2, padding=3)
        self.flops_conv1 = torch.Tensor([49 * h * w * self.inplanes * 3])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        h = conv2d_out_dim(h, kernel_size=3, stride=2, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, padding=1)
        # residual blocks
        self.layer1, h, w = self._make_layer(block, 64 * 1, layers[0], h, w, layer_offset=0)
        self.layer2, h, w = self._make_layer(block, 64 * 2, layers[1], h, w,
                                       stride=2, dilate=replace_stride_with_dilation[0], layer_offset=layers[0])
        self.layer3, h, w = self._make_layer(block, 64 * 4, layers[2], h, w,
                                       stride=2, dilate=replace_stride_with_dilation[1], layer_offset=layers[0] + layers[1])
        self.layer4, h, w = self._make_layer(block, 64 * 8, layers[3], h, w,
                                       stride=2, dilate=replace_stride_with_dilation[2], layer_offset=layers[0] + layers[1] + layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        zero_init_residual = args.zero_init_residual
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # for bn in m.bn2s:
                    #     nn.init.constant_(bn.weight, 0)
                    nn.init.constant_(m.bn2.weight, 0)

    def move_member_vars_to_device(self, target_device):
        if self.args.TDCP_attn and self.args.TDCP_attn_PE:
            for m in self.modules():
                if isinstance(m, PolicyBlock):
                    for _c in ATTENTION_PE_IN_CHANNELS:
                        m.PE_dict[str(_c)] = m.PE_dict[str(_c)].to(target_device)

    def update_shared_net(self, in_planes, out_planes):
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = 3
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim
        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim // 64 * (self.args.gate_channel_ends - self.args.gate_channel_starts)
        keyword = "%d_%d" % (in_dim, out_dim)
        if keyword not in self.gate_fc0s:
            self.gate_fc0s[keyword] = nn.Linear(in_dim, hidden_dim)
            self.gate_fc1s[keyword] = nn.Linear(hidden_dim, out_dim)
            normal_(self.gate_fc0s[keyword].weight, 0, 0.001)
            constant_(self.gate_fc0s[keyword].bias, 0)
            normal_(self.gate_fc1s[keyword].weight, 0, 0.001)
            constant_(self.gate_fc1s[keyword].bias, 0)

    def _make_layer(self, block, planes_list_0, blocks, h, w, stride=1, dilate=False, layer_offset=-1):
        norm_layer = self._norm_layer
        downsample0 = None
        downsample1 = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes_list_0 * block.expansion:
            downsample0 = conv1x1(self.inplanes, planes_list_0 * block.expansion, stride)
            downsample1 = norm_layer(planes_list_0 * block.expansion)

        _d={1:0, 2:1, 4:2, 8:3}
        layer_idx = _d[planes_list_0//64]

        if len(self.args.enabled_layers) > 0:
            enable_policy = layer_offset in self.args.enabled_layers
            print("stage-%d layer-%d (abs: %d) enabled:%s"%(layer_idx, 0, layer_offset, enable_policy))
        elif len(self.args.enabled_stages) > 0:
            enable_policy = layer_idx in self.args.enabled_stages
        else:
            enable_policy = (layer_idx >= self.args.enable_from and layer_idx < self.args.disable_from)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes_list_0, h, w, stride, downsample0, downsample1, self.groups,
                            self.base_width, previous_dilation, norm_layer, shared=(self.gate_fc0s, self.gate_fc1s), shall_enable=enable_policy, args=self.args))
        h = conv2d_out_dim(h, kernel_size=1, stride=stride, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=stride, padding=0)
        self.inplanes = planes_list_0 * block.expansion
        for k in range(1, blocks):

            if len(self.args.enabled_layers) > 0:
                enable_policy = layer_offset + k in self.args.enabled_layers
                print("stage-%d layer-%d (abs: %d) enabled:%s" % (layer_idx, k, layer_offset + k, enable_policy))

            layers.append(block(self.inplanes, planes_list_0, h, w, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, shared=(self.gate_fc0s, self.gate_fc1s), shall_enable=enable_policy, args=self.args))
        return layers, h, w

    def count_flops(self, input_data_shape, **kwargs):
        flops_list = []
        _B, _T, _C, _H, _W = input_data_shape
        input2d_shape = _B*_T, _C, _H, _W

        flops_conv1, data_shape = count_conv2d_flops(input2d_shape, self.conv1)
        data_shape = data_shape[0], data_shape[1], data_shape[2]//2, data_shape[3]//2 #TODO pooling
        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                flops, data_shape = block.count_flops(data_shape, **kwargs)
                flops_list.append(flops)
        return flops_list

    def forward(self, input_data, **kwargs):
        # TODO x.shape (nt, c, h, w)
        frames_num, _, _, _ = input_data.shape

        if "tau" not in kwargs:
            kwargs["tau"] = 1
            kwargs["inline_test"] = True

        mask_stack_list = []  # TODO list for t-dimension
        soft_mask_stack_list = []  # TODO list for t-dimension

        for l_num, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for _, block in enumerate(layers):
                mask_stack_list.append(None)
                soft_mask_stack_list.append(None)

        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        flops = torch.zeros(1, frames_num + 4).to(x.device)
        #+5 for 1) upperbound flops 2) downsample flops 3) channels mask flops
        #       4) orig layer flops (conv1+conv2+downsample)

        idx = 0
        _t = self.args.num_segments
        _b = x.shape[0] // _t

        for li, layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bi, block in enumerate(layers):
                num_layer = 2 * li + bi
                x, mask, mask2, flops, soft_mask = block((x, flops), **kwargs)
                mask_stack_list[idx] = mask
                soft_mask_stack_list[idx] = soft_mask
                idx += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        # The last element is a placeholder for the fully connected flops at the "clip level"
        flops_real = [flops[1:, 0:frames_num].permute(1, 0).contiguous(),
                      self.flops_conv1.to(x.device), torch.Tensor([0.0]).to(x.device)]
        flops_upperbound = flops[1:, -4].unsqueeze(0)
        flops_downsample = flops[1:, -3].unsqueeze(0)
        flops_channels_mask = flops[1:, -2].unsqueeze(0)
        flops_ori  = flops[1:, -1].unsqueeze(0)
        # get outputs
        dyn_outputs = {}
        dyn_outputs["flops_real"] = flops_real
        dyn_outputs["flops_upperbound"] = flops_upperbound
        dyn_outputs["flops_downsample"] = flops_downsample
        dyn_outputs["flops_channels_mask"] = flops_channels_mask
        dyn_outputs["flops_ori"] = flops_ori

        return out, mask_stack_list, dyn_outputs, soft_mask_stack_list


def _resnet_dynstc(arch, block, layers, pretrained, progress, **kwargs):
    model = ResDynSTCNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        # TODO okay now let's load ResNet to DResNet
        model_dict = model.state_dict()
        kvs_to_add = []
        old_to_new_pairs = []
        keys_to_delete = []
        for k in pretrained_dict:
            # TODO layer4.0.downsample.X.weight -> layer4.0.downsampleX.weight
            if "downsample.0" in k:
                old_to_new_pairs.append((k, k.replace("downsample.0", "downsample0")))
            elif "downsample.1" in k:
                old_to_new_pairs.append((k, k.replace("downsample.1", "downsample1")))

        for del_key in keys_to_delete:
            del pretrained_dict[del_key]

        for new_k, new_v in kvs_to_add:
            pretrained_dict[new_k] = new_v

        for old_key, new_key in old_to_new_pairs:
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        for name, layer in model.named_modules():
            if ".bn" in name:
                ly = name[5]
                bb = name[7]
                bn = name[-3] if name[-3] != '_' else name[-4]
                if bn != 'b':
                    layer_name = f"layer{ly}.{bb}.bn{bn}."
                    layer.weight = pretrained_dict[layer_name + "weight"]
                    layer.bias = pretrained_dict[layer_name + "bias"]
                    layer.running_mean = pretrained_dict[layer_name + "running_mean"]
                    layer.running_var = pretrained_dict[layer_name + "running_var"]
    return model


def res18_dynstc_net(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_dynstc('res18_dynstc_net', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)


def res50_dynstc_net(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_dynstc('res50_dynstc_net', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)
