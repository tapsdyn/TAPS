import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--decay_step', '--ds', default=10, type=int)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log', type=str, default='logs')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--exp_header', default="default", type=str, help='experiment header')
parser.add_argument('--rescale_to', default=224, type=int)

# TODO AdaR
parser.add_argument('--reso_list', default=[224], type=int, nargs='+', help="list of resolutions")
parser.add_argument('--eff_loss_after', default=-1, type=int, help="use eff loss after X epochs")
parser.add_argument('--model_paths', default=[], type=str, nargs="+", help='path to load models for backbones')
parser.add_argument('--exp_decay', action='store_true', help="type of annealing")
parser.add_argument('--init_tau', default=5.0, type=float, help="annealing init temperature")
parser.add_argument('--exp_decay_factor', default=-0.045, type=float, help="exp decay factor per epoch")
parser.add_argument('--base_pretrained_from', type=str, default='', help='for base model pretrained path')
parser.add_argument('--skip_training',action='store_true')
parser.add_argument('--random_seed', type=int, default=1007)
parser.add_argument('--train_random_seed', type=int, default=1007)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--real_scsampler', action='store_true')
parser.add_argument('--test_from', type=str, default="")
parser.add_argument('--save_meta', action='store_true')

parser.add_argument('--filelist_suffix', type=str, default="")
parser.add_argument('--frozen_layers', default=[], type=int, nargs="+", help='list of frozen layers')
parser.add_argument('--freeze_corr_bn', action='store_true', help="freeze the corresponding batchnorms")
parser.add_argument('--gate_hidden_dim', type=int, default=16)
parser.add_argument('--gate_all_one_policy', action='store_true')
parser.add_argument('--gate_only_current_policy', action='store_true')
parser.add_argument('--gate_all_zero_policy', action='store_true')
parser.add_argument('--gate_random_hard_policy', action='store_true')
parser.add_argument('--gate_random_soft_policy', action='store_true')
parser.add_argument('--gate_gumbel_use_soft', action='store_true')
parser.add_argument('--gate_bn_between_fcs', '--gbn', action='store_true')
parser.add_argument('--gate_relu_between_fcs', '--grelu', action='store_true')
parser.add_argument('--gate_history', action='store_true')
parser.add_argument('--num_class', default=200, type=int)
parser.add_argument('--gate_stoc_ratio', default=[], type=float, nargs="+") # skip|reuse|keep
parser.add_argument('--gate_history_detach', action='store_true')
parser.add_argument('--gate_history_conv_type', type=str,
                    choices=['None', 'conv1x1', 'conv1x1bnrelu','conv1x1_list', 'conv1x1_res', 'ghost', 'ghostbnrelu'],
                    default='None')
parser.add_argument('--partitions', default=4, type=int)
parser.add_argument('--gate_threshold', action='store_true')
parser.add_argument('--absolute_threshold', default=None, type=float)
parser.add_argument('--relative_max_threshold', default=None, type=float)
parser.add_argument('--relative_keep_threshold', default=None, type=float)
parser.add_argument('--gate_no_skipping', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--not_pin_memory', action='store_true', help='not pin memory')
parser.add_argument('--relative_hidden_size', type=float, default=-1.0)
parser.add_argument('--backup_epoch_list', default=[9, 100], type=int, nargs="+") # keep old best models (for pretraining)
parser.add_argument('--warmup_epochs', default=-1, type=int, help='number of epochs for warmup')
parser.add_argument('--single_linear', action='store_true')
parser.add_argument('--triple_linear', action='store_true')
parser.add_argument('--data_path', default=None, type=str, help='Only use it when your dataset is saved in '
                                                                'a customized path. Otherwise follow the '
                                                                'rules in common_*.py')
parser.add_argument('--exps_path', default=None, type=str, help='Only use it when your experiment log is saved in '
                                                                'a customized path. Otherwise follow the '
                                                                'rules in common_*.py')
parser.add_argument('--granularity', default=1, type=int)
parser.add_argument('--pn_num_outputs', default=3, type=int)
parser.add_argument('--hidden_quota', default=-1, type=int)
parser.add_argument('--always_flip', action='store_true')
parser.add_argument('--conditional_flip', action='store_true')
parser.add_argument('--adaptive_flip', action='store_true')
parser.add_argument('--gate_npb', action='store_true')
parser.add_argument('--gate_lr_factor', default=1.0, type=float)
parser.add_argument('--gate_reduce_type', type=str, default="avg", choices=['avg', 'max'])
parser.add_argument('--enable_from', default=0, type=int)
parser.add_argument('--disable_from', default=4, type=int)
parser.add_argument('--enabled_layers', default=[], type=int, nargs="+")
parser.add_argument('--enabled_stages', default=[], type=int, nargs="+")
parser.add_argument('--gate_channel_starts', default=0, type=int)
parser.add_argument('--gate_channel_ends', default=64, type=int)
parser.add_argument('--zero_init_residual', action='store_true')
parser.add_argument('--save_meta_gate', action='store_true')
parser.add_argument('--skip_log', action='store_true')

# Research
parser.add_argument('--train_folder_suffix', type=str, default="")
parser.add_argument('--val_folder_suffix', type=str, default="")

parser.add_argument('--disable_channelwise_masking', action='store_true')
parser.add_argument('--den_target', default=0.5, type=float, help='target flops retion in pruned network')
parser.add_argument('--sparsity_lambda', default=5, type=float, help='loss weight for network sparsity')
parser.add_argument('--spars_loss_L1', type=bool, default=False)

parser.add_argument('--online_shift', action='store_true', help='use online channels shift in tsm')
parser.add_argument('--shift_copy_pad', action='store_true')
parser.add_argument('--eta_ini', default=1, type=float)
parser.add_argument('--eta_final', default=0, type=float)

parser.add_argument('--TDCP', action='store_true')
parser.add_argument('--TDCP_lstm', action='store_true')
parser.add_argument('--TDCP_blstm', action='store_true')
parser.add_argument('--TDCP_blstm_only', action='store_true')
parser.add_argument('--TDCP_gru', action='store_true')
parser.add_argument('--TDCP_bgru', action='store_true')
parser.add_argument('--TDCP_bgru_only', action='store_true')
parser.add_argument('--TDCP_fc', action='store_true')
parser.add_argument('--TDCP_fc_Sig', action='store_true')
parser.add_argument('--TDCP_skip', action='store_true')
parser.add_argument('--TDCP_soft_mask', action='store_true')
parser.add_argument('--TDCPv2', action='store_true')
parser.add_argument('--TDCPv3', action='store_true')
parser.add_argument('--TDCP_attn', action='store_true')
parser.add_argument('--TDCP_attn_PE', action='store_true')
parser.add_argument('--TDCP_mul_MHA', action='store_true')
parser.add_argument('--TDCP_mul_MHA_disable_ffn', action='store_false')
parser.add_argument('--TDCP_attn_sum', action='store_true')
parser.add_argument('--TDCP_attn_online_inf', action='store_true')
parser.add_argument('--TDCP_attn_LA', type=str, default=None, choices=[None, 'diff', 'SA'])
parser.add_argument('--TDCP_attn_params', default=[64, 1, 256, 16], type=int, nargs='+', help='Taps attention params (embed dim and num heads)')
parser.add_argument('--TDCP_attn_num_heads', default=16, type=int,  help='Taps attention params (embed dim and num heads)')
parser.add_argument('--TDCP_tau', default=1, type=float)
parser.add_argument('--soft_scaling_only', action='store_true')
parser.add_argument('--FBS_scaling_only', action='store_true')

parser.add_argument('--sparsity_warmup_start', default=-1, type=int)
parser.add_argument('--sparsity_warmup_end', default=-1, type=int)
parser.add_argument('--warmup_hard_gates_disable', action='store_true')
parser.add_argument('--hard_gates_disable_outer_mask', action='store_true')
parser.add_argument('--no_gumbel_noise_ft_epoch', default=10**6, type=int)
parser.add_argument('--bs_gamma', default=0.0, type=float, help="batch shaping loss weight")
parser.add_argument('--bs_pdf_alpha', default=0.6, type=float, help="target fraction of 1's in masks")
parser.add_argument('--bs_pdf_beta', default=0.4, type=float, help="target fraction of 0's in masks")
parser.add_argument('--bs_beta_cdf_res', default=256, type=int, help="resolution in cdf lut of batch shaping loss")
parser.add_argument('--bs_loss_start_epoch', default=0, type=int,
                    help="batch shaping loss will be applied and linearly annealed to 0 starting from this epoch")
parser.add_argument('--bs_loss_stop_epoch', default=0, type=int,
                    help="batch shaping loss will be linearly annealed to 0 up to this epoch")
parser.add_argument('--no_soft_mask_until_epoch', default=-1, type=int,
                    help="channel scaling is disabled until this epoch. typically should be same as bs_loss_stop_epoch")
parser.add_argument('--bs_policy_in_attached', action='store_true')
parser.add_argument('--bs_from_block', default=0, type=int, help='apply batch shaping loss only from this backbone'
                                                                 ' block and onwards')
