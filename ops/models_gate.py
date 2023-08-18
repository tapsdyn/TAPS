from torch import nn
import sys
# print(sys.path)
sys.path.append('/usr/local/lib/python3.7/site-packages')
from ops.transforms import *
from torch.nn.init import normal_, constant_
import ops.res_dynstc_net as res_dynstc_net


def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell


class TSN_Gate(nn.Module):
    def __init__(self, args):
        super(TSN_Gate, self).__init__()
        self.args = args
        self.num_segments = args.num_segments
        self.num_class = args.num_class

        self.base_model = self._prepare_base_model()
        self.new_fc, self.flops_fc = self._prepare_fc(args.num_class)
        self._enable_pbn = False

    def _prep_a_net(self, model_name, shall_pretrain):
        if "dynstc" in model_name:
            model = getattr(res_dynstc_net, model_name)(shall_pretrain, args=self.args)
            model.last_layer_name = 'fc'
        else:
            exit("I don't how to prep this net:%s; see models_gate.py:: _prep_a_net" % model_name)
        return model

    def _prepare_base_model(self):
        shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
        model = self._prep_a_net(self.args.arch, shall_pretrain)

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        return model

    def _prepare_fc(self, num_class):
        def make_a_linear(input_dim, output_dim):
            linear_model = nn.Linear(input_dim, output_dim)
            normal_(linear_model.weight, 0, 0.001)
            constant_(linear_model.bias, 0)
            return linear_model

        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        new_fc = make_a_linear(feature_dim, num_class)
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.args.dropout))
        flops_fc = feature_dim * num_class + num_class
        return new_fc, flops_fc

    def move_model_member_vars_to_device(self, target_device):
        self.base_model.move_member_vars_to_device(target_device)

    def forward(self, *argv, **kwargs):
        input_data = kwargs["input"][0]
        is_training = kwargs["is_training"]
        curr_step =  kwargs["curr_step"]
        epoch = kwargs["epoch"]
        is_first_batch = kwargs["first_batch"]
        disable_hard_channels_mask = False
        if self.args.warmup_hard_gates_disable and epoch < self.args.sparsity_warmup_start:
            disable_hard_channels_mask = True
        disable_soft_channels_mask = False
        if epoch < self.args.no_soft_mask_until_epoch:
            disable_soft_channels_mask = True
        no_gumbel_noise_fine_tune = False
        if epoch >= self.args.no_gumbel_noise_ft_epoch:
            no_gumbel_noise_fine_tune = True

        _b, _tc, _h, _w = input_data.shape
        _t, _c = _tc // 3, 3
        if "tau" not in kwargs:
            kwargs["tau"] = None

        if is_first_batch:
            self.move_model_member_vars_to_device(target_device=input_data.device)

        feat, mask_stack_list, dyn_outputs, soft_mask_stack_list =\
            self.base_model(input_data.view(_b * _t, _c, _h, _w),
                            tau=kwargs["tau"],
                            is_training=is_training,
                            curr_step=curr_step,
                            fc_weights=self.new_fc.weight,
                            targets=kwargs['targets'],
                            disable_hard_channels_mask=disable_hard_channels_mask,
                            disable_soft_channels_mask=disable_soft_channels_mask,
                            no_gumbel_noise_fine_tune=no_gumbel_noise_fine_tune)
        batch_size = _b
        frames_per_clip = _t
        dyn_outputs["flops_real"][0] = dyn_outputs["flops_real"][0].view(batch_size, frames_per_clip, -1) # -> [BatchSize, FramesPerClip, #ResNetLayers]: The actual measured flops per frame and residual block
        dyn_outputs["flops_real"][2] = torch.Tensor([self.flops_fc]).to(dyn_outputs["flops_real"][0].device) # -> [scalar]: flops (per clip) of the FC layer (always static)
        # dyn_outputs["flops_real"][1] : [scalar]: flops (per 1 frame) of the FIRST CONVOLUTION in the network (always static)
        # dyn_outputs["flops_ori"] : [1, #ResNetLayers]: flops (per clip) of the STATIC model ops (conv1+conv2+downsampling+channels policy) at each ResNetLayer
        # dyn_outputs["flops_upperbound"] : [1, #ResNetLayers]: flops (per clip) at each ResNetLayer, considering only downstrram (2nd conv in ResNet block) flops saving
        # dyn_outputs["flops_downsample"]: [1, #ResNetLayers]: flops (per clip) of the downsampling (if exists) at each ResNetLayer
        # dyn_outputs["flops_channels_mask"]: [1, #ResNetLayers]: flops (per clip) of the channels policy net at each ResNetLayer

        base_out = self.new_fc(feat.view(_b * _t, -1)).view(_b, _t, -1)
        output = base_out.mean(dim=1).squeeze(1)

        if self.args.save_meta_gate:
            return output, mask_stack_list, None, torch.stack([base_out], dim=1), dyn_outputs
        else:
            return output, mask_stack_list, None, torch.stack([base_out], dim=1), dyn_outputs, soft_mask_stack_list


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Gate, self).train(mode)
        if mode and len(self.args.frozen_layers) > 0 and self.args.freeze_corr_bn:
            for layer_idx in self.args.frozen_layers:
                for km in self.base_model.named_modules():
                    k, m = km
                    if layer_idx == 0:
                        if "bn1" in k and "layer" not in k and (
                                isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                    else:
                        if "layer%d" % (layer_idx) in k and (
                                isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                            m.eval()
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        gate_ops_weight = []
        gate_ops_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for m_name, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())

                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d) and ("gate_bn" in m_name or "attention" in m_name):
                if not self.args.gate_npb:
                    bn.extend(list(m.parameters()))

            elif self.args.gate_lr_factor != 1 and "gate_fc" in m_name and isinstance(m, torch.nn.Linear):
                assert(isinstance(m, torch.nn.Linear) == False)
                ps = list(m.parameters())
                gate_ops_weight.append(ps[0])
                if len(ps) == 2:
                    gate_ops_bias.append(ps[1])

            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.LSTMCell) or isinstance(m, torch.nn.LSTM) or isinstance(m, torch.nn.GRU):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for gate policy
            {'params': gate_ops_weight, 'lr_mult': self.args.gate_lr_factor, 'decay_mult': 1,
             'name': "gate_ops_weight"},
            {'params': gate_ops_bias, 'lr_mult': self.args.gate_lr_factor * 2, 'decay_mult': 0},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
