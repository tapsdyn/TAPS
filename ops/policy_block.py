import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_
import math
import numpy as np

ATTENTION_PE_IN_CHANNELS = [64, 128, 256, 512, 1024, 2048]


class PolicyBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer, shared, args):
        super(PolicyBlock, self).__init__()
        self.args = args
        self.norm_layer = norm_layer
        self.shared = shared
        in_factor = 1
        out_factor = 2
        if self.args.gate_history:
            in_factor = 2
            if not self.args.gate_no_skipping:
                out_factor = self.args.pn_num_outputs
        if self.args.pn_num_outputs == 1:
            out_factor = 1
        if self.args.TDCP_skip:
            in_factor = 2
        if self.args.TDCP_blstm or self.args.TDCP_bgru or self.args.TDCP_blstm_only or self.args.TDCP_bgru_only:
            in_factor = in_factor + 1
        if self.args.TDCP_attn:
            in_factor = in_factor + 1
            if self.args.TDCP_attn_LA is not None:
                in_factor = in_factor + 1
        self.action_dim = out_factor
        self.flops = 0.0 # flops calculation supported only for 2-layer fully connected network

        in_dim = in_planes * in_factor
        out_dim = out_planes * out_factor // self.args.granularity
        out_dim = out_dim * (args.gate_channel_ends - args.gate_channel_starts) // 64
        self.num_channels = out_dim // self.action_dim

        keyword = "%d_%d" % (in_dim, out_dim)
        if self.args.relative_hidden_size > 0:
            hidden_dim = int(self.args.relative_hidden_size * out_planes // self.args.granularity)
        elif self.args.hidden_quota > 0:
            hidden_dim = self.args.hidden_quota // (out_planes // self.args.granularity)
        else:
            hidden_dim = self.args.gate_hidden_dim

        if self.args.single_linear:
            self.gate_fc0 = nn.Linear(in_dim, out_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(out_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
        elif self.args.triple_linear:
            self.gate_fc0 = nn.Linear(in_planes * in_factor, hidden_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            self.gate_fc1 = nn.Linear(hidden_dim, hidden_dim)

            if self.args.gate_bn_between_fcs:
                self.gate_bn1 = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu1 = nn.ReLU(inplace=True)
            self.gate_fc2 = nn.Linear(hidden_dim, out_dim)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)
            normal_(self.gate_fc2.weight, 0, 0.001)
            constant_(self.gate_fc2.bias, 0)

        else:
            self.gate_fc0 = nn.Linear(in_dim, hidden_dim)
            self.flops = self.get_flops(in_dim, hidden_dim, out_dim)
            if self.args.gate_bn_between_fcs:
                self.gate_bn = nn.BatchNorm1d(hidden_dim)
            if self.args.gate_relu_between_fcs:
                self.gate_relu = nn.ReLU(inplace=True)

            self.gate_fc1 = nn.Linear(hidden_dim, out_dim)

            normal_(self.gate_fc0.weight, 0, 0.001)
            constant_(self.gate_fc0.bias, 0)
            normal_(self.gate_fc1.weight, 0, 0.001)
            constant_(self.gate_fc1.bias, 0)

            if self.args.TDCP_fc:
                self.TDCP_fc = nn.Sequential(
                    nn.Linear(in_planes, in_planes),
                    nn.ReLU(inplace=True)
                )
            elif self.args.TDCP_fc_Sig:
                self.TDCP_fc = nn.Sequential(
                    nn.Linear(in_planes, in_planes),
                    nn.ReLU(inplace=True)
                )

            if self.args.TDCP_lstm or self.args.TDCP_blstm or self.args.TDCP_blstm_only:
                self.TDCP_rnn = nn.LSTM(input_size=in_planes, hidden_size=in_planes, batch_first=True, bidirectional=(self.args.TDCP_blstm or self.args.TDCP_blstm_only))
            elif self.args.TDCP_gru or self.args.TDCP_bgru or self.args.TDCP_bgru_only:
                self.TDCP_rnn = nn.GRU(input_size=in_planes, hidden_size=in_planes, batch_first=True, bidirectional=(self.args.TDCP_bgru or self.args.TDCP_bgru_only))
            elif self.args.TDCP_attn:
                self.TDCP_sa = nn.MultiheadAttention(embed_dim=in_planes, num_heads=1, batch_first=True)
                self.TDCP_ca = nn.MultiheadAttention(embed_dim=in_planes, num_heads=self.args.TDCP_attn_num_heads, batch_first=True)
                self.TDCP_ca2 = nn.MultiheadAttention(embed_dim=in_planes, num_heads=self.args.TDCP_attn_num_heads, batch_first=True)
                self.TDCP_ffn = nn.Sequential(
                    nn.Linear(in_planes, in_planes * 4),  # Replace hidden_dim with desired size
                    nn.ReLU(),
                    nn.Linear(in_planes * 4, in_planes)
                )
                self.TDCP_norm1 = nn.LayerNorm(in_planes)
                self.TDCP_norm2 = nn.LayerNorm(in_planes)
                self.TDCP_norm3 = nn.LayerNorm(in_planes)

                if self.args.TDCP_attn_PE:
                    self.PE_dict = {}
                    _t = self.args.num_segments
                    for _c in ATTENTION_PE_IN_CHANNELS:
                        PE = torch.zeros(1, _t, _c)  # [1, t, c]
                        position = torch.arange(0, _t).unsqueeze(1)
                        div_term = torch.exp(
                            torch.arange(0, _c, 2) * -(math.log(10000.0) / _c))  # 1/e^(100*2i/_c)), 0<i<_c/2
                        PE[0, :, 0::2] = torch.sin(position * div_term)
                        PE[0, :, 1::2] = torch.cos(position * div_term)
                        self.PE_dict[str(_c)] = PE.expand(self.args.batch_size, -1, -1)
        if out_factor == 1:
            self.binary_gate = BinConcrete()
            self.sigmoid = nn.Sigmoid()

    def get_flops(self, in_dim, hidden_dim, out_dim):
        return hidden_dim * in_dim + out_dim * hidden_dim

    def forward(self, x, frame_num=None, **kwargs):
        # data preparation
        x_input = x

        if self.args.gate_reduce_type=="avg":
            x_c = nn.AdaptiveAvgPool2d((1, 1))(x_input)
        elif self.args.gate_reduce_type=="max":
            x_c = nn.AdaptiveMaxPool2d((1, 1))(x_input)
        x_c = torch.flatten(x_c, 1)
        _nt, _c = x_c.shape
        _t = self.args.num_segments
        _n = _nt // _t

        # Attention
        if self.args.TDCP_attn:
            attn_mask = torch.triu(torch.ones(_t, _t, device=x_c.device), diagonal=1) if self.args.TDCP_attn_online_inf else None
            x_c_sa, x_c_ca = torch.zeros_like(x_c), torch.zeros_like(x_c)  # [nt, c]
            x_c_reshape = x_c.view(_n, _t, _c)  # [n, t, c]

            # Local Attention
            if self.args.TDCP_attn_LA == 'diff':
                h_vec, f_vec = x_c_reshape.clone(), x_c_reshape.clone()  # [n, t, c]
                h_vec[:, 1:, :] = h_vec[:, :-1, :].clone()  # [n, t, c]
                f_vec[:, :-1, :] = f_vec[:, 1:, :].clone()  # [n, t, c]
                x_c_sa = torch.abs(x_c_reshape - h_vec) + torch.abs(x_c_reshape - f_vec)  # [n, t, c]
                x_c_sa = x_c_sa.view(_nt, _c)
            elif self.args.TDCP_attn_LA == 'SA': # using simple LA
                x_c_sa, _ = self.TDCP_sa(x_c, x_c, x_c)  # [nt, c]
            else:
                x_c_sa = None

            # GLobal attention
            if self.args.TDCP_attn_PE:
                x_c_PE = self.PE_dict[str(_c)]
                if x_c_PE.shape[0] == _n:
                    x_c_ca, _ = self.TDCP_ca(x_c_reshape + x_c_PE,  x_c_reshape + x_c_PE, x_c_reshape, attn_mask=attn_mask)  # [n, t, c]
                else:  # Profiling
                    x_c_ca, _ = self.TDCP_ca(x_c_reshape, x_c_reshape, x_c_reshape, attn_mask=attn_mask)  # [n, t, c]

            else:
                x_c_ca, _ = self.TDCP_ca(x_c_reshape,  x_c_reshape, x_c_reshape, attn_mask=attn_mask)  # [n, t, c]

            if self.args.TDCP_mul_MHA:
                residual1 = x_c_reshape + x_c_ca  # Add residual connection
                norm_output1 = self.TDCP_norm1(residual1)  # Apply layer normalization

                # Second MHA layer
                attn_output1, _ = self.TDCP_ca2(norm_output1, norm_output1, norm_output1, attn_mask=attn_mask)
                residual2 = norm_output1 + attn_output1
                norm_output2 = self.TDCP_norm2(residual2)

                # Third MHA layer
                if self.args.TDCP_mul_MHA_disable_ffn:
                    ffn_output = self.TDCP_ffn(norm_output2)
                    residual3 = norm_output2 + ffn_output
                    norm_output3 = self.TDCP_norm3(residual3)
                    x_c_ca = norm_output3
                else:
                    x_c_ca = norm_output2

            if self.args.TDCP_attn_sum:
                x_c_ca = x_c_ca.mean(dim=1).expand(-1, _t, -1)

            x_c_ca = x_c_ca.contiguous().view(_nt, _c)  # [nt, c]

            if self.args.TDCP_attn_LA is not None:
                x_c = torch.cat([x_c_sa, x_c_ca, x_c], dim=-1) # [3 * nt, c]
            else:
                x_c = torch.cat([x_c_ca, x_c], dim=-1) # [2 * nt, c]

        # BLSTM only
        if self.args.TDCP_blstm_only or self.args.TDCP_bgru_only:
            x_c_reshape = x_c.view(_n, _t, _c)
            x_c_reshape = self.TDCP_rnn(x_c_reshape)[0]
            x_c = x_c_reshape.contiguous().view(_nt, 2 * _c)

        # history
        if self.args.gate_history or self.args.TDCP_skip:
            # CA - fc and Sigmoid
            if self.args.TDCP_fc:
                x_c = self.TDCP_fc(x_c)
            x_c_reshape = x_c.view(_n, _t, _c)

            if not (self.args.TDCPv2 or self.args.TDCPv3):
                h_vec = torch.zeros_like(x_c_reshape)
                h_vec[:, 1:] = x_c_reshape[:, :-1]
                if self.args.TDCP:
                    h_vec = h_vec.view(_n, _t, _c)
                    if self.args.TDCP_lstm or self.args.TDCP_blstm or self.args.TDCP_gru or self.args.TDCP_bgru:
                        h_vec = self.TDCP_rnn(h_vec)[0]
                        h_vec = h_vec.contiguous()
                        if self.args.TDCP_blstm or self.args.TDCP_bgru:
                            _c = _c * 2
                    else:
                        for t in range(1, _t):
                            h_vec[:, t, :] = self.args.TDCP_tau * h_vec[:, t, :] + (1 - self.args.TDCP_tau) * h_vec[:, t - 1, :]
            else:
                h_vec_copy = self.TDCP_rnn(x_c_reshape)[0]
                h_vec = torch.zeros_like(h_vec_copy)
                if self.args.TDCPv3:
                    h_vec[:, 1:, :_c] = h_vec_copy[:, :-1, :_c].clone()
                    h_vec[:, :-1, _c:] = h_vec_copy[:, 1:, _c:].clone()
                h_vec = h_vec.contiguous()
                if self.args.TDCP_blstm or self.args.TDCP_bgru:
                    _c = _c * 2
            h_vec = h_vec.view(_nt, _c)
            x_c = torch.cat([h_vec, x_c], dim=-1)

        # fully-connected embedding
        if self.args.single_linear:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)

        elif self.args.triple_linear:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)
            x_c = self.gate_fc1(x_c)

            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn1(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu1(x_c)
            x_c = self.gate_fc2(x_c)

        else:
            x_c = self.gate_fc0(x_c)
            if self.args.gate_bn_between_fcs:
                x_c = x_c.unsqueeze(-1)
                x_c = self.gate_bn(x_c)
                x_c = x_c.squeeze(-1)
            if self.args.gate_relu_between_fcs:
                x_c = self.gate_relu(x_c)
            x_c = self.gate_fc1(x_c)

        # gating operations
        x_c2d = x_c.view(x.shape[0], self.num_channels // self.args.granularity, self.action_dim)
        soft_mask = None
        if self.args.pn_num_outputs in [2, 3, 4]:
            x_c2d = torch.log(F.softmax(x_c2d, dim=2).clamp(min=1e-8))
            mask = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=not self.args.gate_gumbel_use_soft)
            if self.args.TDCP_soft_mask:
                # soft_mask = F.gumbel_softmax(logits=x_c2d, tau=kwargs["tau"], hard=False)
                soft_mask = torch.softmax(x_c2d, dim=2)
        elif self.args.pn_num_outputs == 1:
            # x_c2d, mask, soft_mask: [BatchSize, NumOutputChannels, 1]
            no_gumbel_noise_fine_tune = kwargs['no_gumbel_noise_fine_tune'] if 'no_gumbel_noise_fine_tune' in kwargs\
                else False
            mask = self.binary_gate(x_c2d, kwargs['tau'], gumbel_noise=not no_gumbel_noise_fine_tune)
            if self.args.TDCP_soft_mask:
                soft_mask = self.sigmoid(x_c2d)
        else:
            mask = F.sigmoid(x_c2d * 3)
            with torch.no_grad():
                mask = torch.round(mask)
                # div = torch.tensor(torch.logical_not(torch.logical_xor(mask[:, :, 0], mask[:, :, 1])) + 1, device=mask.device).unsqueeze(2)
                div = torch.Tensor(np.logical_not(np.logical_xor(mask[:, :, 0].cpu().numpy(), mask[:, :, 1].cpu().numpy())) + 1).unsqueeze(2)
                mask = mask / div.to(mask.device)

        if self.args.soft_scaling_only:
            outer_mask = torch.zeros_like(mask)
            outer_mask[:, :, -1] = 1.
            return outer_mask, soft_mask

        if self.args.granularity>1:
            mask = mask.repeat(1, self.args.granularity, 1)
        if self.args.gate_channel_starts>0 or self.args.gate_channel_ends<64:
            full_channels = mask.shape[1] // (self.args.gate_channel_ends-self.args.gate_channel_starts) * 64
            channel_starts = full_channels // 64 * self.args.gate_channel_starts
            channel_ends = full_channels // 64 * self.args.gate_channel_ends
            outer_mask = torch.zeros(mask.shape[0], full_channels, mask.shape[2]).to(mask.device)
            outer_mask[:, :, -1] = 1.
            outer_mask[:, channel_starts:channel_ends] = mask

            return outer_mask
        else:
            return mask, soft_mask  # TODO: BT*C*ACT_DIM


class BinConcrete(nn.Module):
    '''
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
    '''
    def __init__(self, eps=1e-8):
        super(BinConcrete, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=0.66667, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard


def handcraft_policy_for_masks(x, out, num_channels, use_current, args):
    factor = 3 if args.gate_history else 2

    if use_current:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
        mask[:, :, -1] = 1.

    elif args.gate_all_zero_policy:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_all_one_policy:
        mask = torch.ones(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_only_current_policy:
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)
        mask[:, :, -1] = 1.
    elif args.gate_random_soft_policy:
        mask = torch.rand(x.shape[0], num_channels, factor, device=x.device)
    elif args.gate_random_hard_policy:
        tmp_value = torch.rand(x.shape[0], num_channels, device=x.device)
        mask = torch.zeros(x.shape[0], num_channels, factor, device=x.device)

        if len(args.gate_stoc_ratio) > 0:
            _ratio = args.gate_stoc_ratio
        else:
            _ratio = [0.333, 0.333, 0.334] if args.gate_history else [0.5, 0.5]
        mask[:, :, 0][torch.where(tmp_value < _ratio[0])] = 1
        if args.gate_history:
            mask[:, :, 1][torch.where((tmp_value < _ratio[1] + _ratio[0]) & (tmp_value > _ratio[0]))] = 1
            mask[:, :, 2][torch.where(tmp_value > _ratio[1] + _ratio[0])] = 1

    elif args.gate_threshold:
        stat = torch.norm(out, dim=[2, 3], p=1) / out.shape[2] / out.shape[3]
        mask = torch.ones_like(stat).float()
        if args.absolute_threshold is not None:
            mask[torch.where(stat < args.absolute_threshold)] = 0
        else:
            if args.relative_max_threshold is not None:
                mask[torch.where(
                    stat < torch.max(stat, dim=1)[0].unsqueeze(-1) * args.relative_max_threshold)] = 0
            else:
                mask = torch.zeros_like(stat)
                c_ids = torch.topk(stat, k=int(mask.shape[1] * args.relative_keep_threshold), dim=1)[1]  # TODO B*K
                b_ids = torch.tensor([iii for iii in range(mask.shape[0])]).to(mask.device).unsqueeze(-1).expand(c_ids.shape)  # TODO B*K
                mask[b_ids.detach().flatten(), c_ids.detach().flatten()] = 1

        mask = torch.stack([1 - mask, mask], dim=-1)

    return mask
