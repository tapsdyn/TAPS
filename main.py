import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import shutil
import multiprocessing
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from ops.dataset import TSNDataSet
from ops.models_gate import TSN_Gate
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder, \
    init_gflops_table, compute_gflops_by_mask, compute_gflops_dynstc, adjust_learning_rate, ExpAnnealing
from opts import parser
from ops.my_logger import Logger
import numpy as np
import common
from os.path import join as ospj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from shutil import copyfile
from regularization import Loss
import pickle


def inner_main(argv):
    args = parser.parse_args()
    common.set_manual_data_path(args.data_path, args.exps_path)

    test_mode = (args.test_from != "")

    set_random_seed(args.random_seed, args)

    args.num_class, args.train_list, args.val_list, args.root_path, prefix, args.train_folder_suffix, args.val_folder_suffix = dataset_config.return_dataset(args.dataset, args.data_path)

    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    logger = Logger()
    sys.stdout = logger

    exp_header_list_full = np.asarray([f for f in os.listdir(common.EXPS_PATH) if f.startswith("g")])
    exp_header_list = np.asarray([f[f.find("_") + 1:] for f in os.listdir(common.EXPS_PATH) if f.startswith("g")])
    resume_recorders_pkl = ''
    if args.exp_header in exp_header_list:
        earlier_exp_folders = exp_header_list_full[np.where(exp_header_list == args.exp_header)]
        # sort by time
        earlier_exp_folders = np.sort(earlier_exp_folders)
        num_of_eralier_folders = len(earlier_exp_folders)
        for early_folder_ind in range(num_of_eralier_folders - 1, -1, -1):
            exp_full_name = earlier_exp_folders[early_folder_ind]
            exp_path = os.path.join(common.EXPS_PATH, exp_full_name)

            # if there are no saved models - delete the folder and continue
            if len(os.listdir(os.path.join(exp_path, "models"))) == 0:
                shutil.rmtree(exp_path)
            # retrieve the latest ckpt
            else:
                args.base_pretrained_from = f"{exp_full_name}/models/ckpt.latest.pth.tar"
                log_fn = [f for f in os.listdir(exp_path) if "log" in f][0]
                with open(os.path.join(exp_path, log_fn), "r") as last_exp:
                    lines_string = last_exp.read()
                    # last_epoch = lines_string.count("Testing") AR: Failed in case of double resume.
                    last_epoch_start_ind = lines_string.rfind("Epoch:[") + len("Epoch:[")
                    last_epoch_end_ind = last_epoch_start_ind + lines_string[last_epoch_start_ind:].find("]")
                    last_epoch = int(lines_string[last_epoch_start_ind:last_epoch_end_ind])
                args.start_epoch = last_epoch
                # check if validation recordings can be restored (for best top1 + flops tracking)
                last_saved_recorders_pkl = os.path.join(exp_path, 'val_records.pkl')
                if os.path.exists(last_saved_recorders_pkl):
                    resume_recorders_pkl = last_saved_recorders_pkl
                break

    model = TSN_Gate(args=args)

    use_DCP = not args.disable_channelwise_masking
    base_model_gflops, gflops_list, g_meta = init_gflops_table(model, args)
    if test_mode:
        args.warmup_hard_gates_disable = False
        args.no_soft_mask_until_epoch = -1

    policies = model.get_optim_policies()
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.gpus is None:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cpu()
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if test_mode or args.base_pretrained_from != "":
        the_model_path = args.base_pretrained_from
        if test_mode:
            if "pth.tar" not in args.test_from:
                the_model_path = ospj(args.test_from, "models", "ckpt.best.pth.tar")
            else:
                the_model_path = args.test_from
        the_model_path = common.EXPS_PATH + "/" + the_model_path
        if True:#args.gpus is None: # solves some OOM (https://discuss.pytorch.org/t/resuming-training-raises-error-cuda-out-of-memory/78987/5)
            sd = torch.load(the_model_path, map_location=torch.device('cpu'))['state_dict']
        else:
            sd = torch.load(the_model_path)['state_dict']
        model_dict = model.state_dict()
        model_dict.update(sd)
        model.load_state_dict(model_dict, strict=False)

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(model, prefix, args)
    if args.gpus is None:
        criterion = torch.nn.CrossEntropyLoss().cpu()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    loss_device = torch.device('cpu') if args.gpus is None else torch.device('cuda:0')
    my_criterion = Loss(args.bs_pdf_alpha, args.bs_pdf_beta, args.bs_beta_cdf_res, loss_device)

    exp_full_path = setup_log_directory(args.exp_header, test_mode, args, logger)

    if not test_mode:
        with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
            f.write(str(args))

    init_empty_recorders = True
    if len(resume_recorders_pkl) > 0:
        try:
            with open(resume_recorders_pkl, 'rb') as f:
                [map_record, mmap_record, prec_record, prec5_record, gflops_record] = pickle.load(f)
            init_empty_recorders = False
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print('Could not load recorders data for resumed training')
    if init_empty_recorders:
        map_record, mmap_record, prec_record, prec5_record, gflops_record = get_recorders(5)

    best_train_usage_str = None
    best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        if use_DCP and epoch < args.eff_loss_after:
            args.disable_channelwise_masking = True
        elif use_DCP and epoch >= args.eff_loss_after:
            args.disable_channelwise_masking = False

        # train for one epoch
        if not args.skip_training and not test_mode:
            set_random_seed(args.train_random_seed + epoch, args)
            adjust_learning_rate(optimizer, epoch, -1, -1, args.lr_type, args.lr_steps, args)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list,
                                    g_meta, my_criterion, args)
        else:
            train_usage_str = "(Eval mode)"

        torch.cuda.empty_cache()

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed, args)
            mAP, mmAP, prec1, prec5, val_usage_str, gflops_per_clip = \
                validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path,
                         my_criterion, args)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)
            prec5_record.update(prec5)
            gflops_record.update(float(gflops_per_clip.cpu().numpy()))
            # save records for loading if training resumes
            with open(os.path.join(exp_full_path, 'val_records.pkl'), 'wb') as f:
                pickle.dump([map_record, mmap_record, prec_record, prec5_record, gflops_record], f)

            if prec_record.is_current_best():
                best_train_usage_str = train_usage_str if not args.skip_training else "(Eval Mode)"
                best_val_usage_str = val_usage_str

            print('Best Prec@1: %.3f (epoch=%d) w. Prec@5: %.3f' % (
                prec_record.best_val, prec_record.best_at,
                prec5_record.at(prec_record.best_at)))

            if test_mode or args.skip_training:  # only runs for one epoch
                break
            else:
                saved_things = {'state_dict': model.state_dict()}
                save_checkpoint(saved_things, prec_record.is_current_best(), False, exp_full_path, "ckpt.best")
                save_checkpoint(saved_things, True, False, exp_full_path, "ckpt.latest")

                if epoch in args.backup_epoch_list:
                    save_checkpoint(None, False, True, exp_full_path, str(epoch))
                torch.cuda.empty_cache()

    # after fininshing all the epochs
    if test_mode:
        if args.skip_log == False:
            os.rename(logger._log_path, ospj(logger._log_dir_name, logger._log_file_name[:-4] +
                                             "_mm_%.2f_a_%.2f_f.txt" % (mmap_record.best_val, prec_record.best_val)))
    else:
        print("Best train usage:%s\nBest val usage:%s" % (best_train_usage_str, best_val_usage_str))


def build_dataflow(dataset, is_train, batch_size, workers, not_pin_memory):
    workers = min(workers, multiprocessing.cpu_count())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                              num_workers=workers, pin_memory=not not_pin_memory, sampler=None,
                                              drop_last=is_train)
    return data_loader


def get_data_loaders(model, prefix, args):
    train_transform_flip = torchvision.transforms.Compose([
        model.module.get_augmentation(flip=True),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    train_transform_nofl = torchvision.transforms.Compose([
        model.module.get_augmentation(flip=False),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    val_transform = torchvision.transforms.Compose([
        GroupScale(int(model.module.scale_size)),
        GroupCenterCrop(model.module.crop_size),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(model.module.input_mean, model.module.input_std),
    ])

    train_dataset = TSNDataSet(args.root_path, args.train_list,
                               num_segments=args.num_segments,
                               image_tmpl=prefix,
                               transform=(train_transform_flip, train_transform_nofl),
                               dense_sample=args.dense_sample,
                               dataset=args.dataset,
                               filelist_suffix=args.filelist_suffix,
                               folder_suffix=args.train_folder_suffix,
                               save_meta=args.save_meta,
                               always_flip=args.always_flip,
                               conditional_flip=args.conditional_flip,
                               adaptive_flip=args.adaptive_flip)

    val_dataset = TSNDataSet(args.root_path, args.val_list,
                             num_segments=args.num_segments,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=(val_transform, val_transform),
                             dense_sample=args.dense_sample,
                             dataset=args.dataset,
                             filelist_suffix=args.filelist_suffix,
                             folder_suffix=args.val_folder_suffix,
                             save_meta=args.save_meta)

    train_loader = build_dataflow(train_dataset, True, args.batch_size, args.workers, args.not_pin_memory)
    val_loader = build_dataflow(val_dataset, False, args.batch_size, args.workers, args.not_pin_memory)

    return train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch, base_model_gflops, gflops_list, g_meta, my_criterion,
          args):
    batch_time, data_time, closses, rlosses, bslosses, losses, top1, top5 = get_average_meters(8)

    mask_stack_list_list = [0 for _ in gflops_list]
    upb_batch_gflops_list = []
    real_batch_gflops_list = []

    tau = get_current_temperature(epoch, args.exp_decay, args.init_tau, args.exp_decay_factor)

    # switch to train mode
    model.module.partialBN(not args.no_partialbn)
    model.train()

    end = time.time()
    print("#%s# lr:%.6f\ttau:%.4f" % (args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau))

    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.warmup_epochs > 0:
            adjust_learning_rate(optimizer, epoch, len(train_loader), i, "linear", None, args)

        # input and target
        batchsize = input_tuple[0].size(0)

        if args.gpus is None:
            input_var_list = [input_item.cpu() for input_item in input_tuple[:-1]]
            target = input_tuple[-1].cpu()
        else:
            input_var_list = [input_item.cuda(non_blocking=True) for input_item in input_tuple[:-1]]
            target = input_tuple[-1].cuda(non_blocking=True)

        # model forward function & measure losses and accuracy
        output, mask_stack_list, _, _, dyn_outputs, soft_mask_stack_list = \
            model(input=input_var_list, tau=tau, is_training=True, curr_step=epoch * len(train_loader) + i,
                  targets=target, epoch=epoch, first_batch=i == 0)

        upb_gflops_dynstc, real_gflops_dynstc = compute_gflops_dynstc(dyn_outputs, args.num_segments,
                                                                      args.batch_size, base_model_gflops)

        closs, rloss, bs_loss = my_criterion(output, target[:, 0], dyn_outputs['flops_real'],
                                             dyn_outputs['flops_ori'][0], args.batch_size, args.den_target,
                                             args.sparsity_lambda, soft_mask_stack_list, epoch, args)

        upb_batch_gflops_list.append(upb_gflops_dynstc.detach())
        real_batch_gflops_list.append(real_gflops_dynstc.detach())

        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

        dynstc_loss = closs.mean() + rloss.mean() + bs_loss.mean()

        closses.update(closs.mean().item(), batchsize)
        rlosses.update(rloss.mean().item(), batchsize)
        bslosses.update(bs_loss.mean().item(), batchsize)
        losses.update(dynstc_loss.item(), batchsize)

        top1.update(prec1.item(), batchsize)
        top5.update(prec5.item(), batchsize)

        # compute gradient and do SGD step
        dynstc_loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        optimizer.zero_grad()

        # gather masks
        for layer_i, mask_stack in enumerate(mask_stack_list):
            mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] lr {3:.6f} '
                            'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), optimizer.param_groups[-1]['lr'] * 0.1, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))  # TODO

            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Cls', loss=closses)
            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Spar', loss=rlosses)
            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Bshp', loss=bslosses)

            print(print_output)

    upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
    real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
    print(usage_str)

    return usage_str


def validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path, my_criterion,
             args):
    batch_time, closses, rlosses, bslosses, losses, top1, top5 = get_average_meters(7)
    all_results = []
    all_targets = []

    tau = get_current_temperature(epoch, args.exp_decay, args.init_tau, args.exp_decay_factor)

    mask_stack_list_list = [0 for _ in gflops_list]
    mask_stack_list_skipping = [0 for _ in gflops_list]
    upb_batch_gflops_list = []
    real_batch_gflops_list = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        mask_t1_sum_S = []
        mask_t1_sum_R = []
        mask_t1_sum_K = []
        mask_t1_sum_M = []
        mask_s_sum = []
        # stats per channel
        mask_t1_S_bins = []
        mask_t1_R_bins = []
        mask_t1_K_bins = []
        for num_layer in range(len(gflops_list)):
            mask_t1_S_bins.append([])
            mask_t1_K_bins.append([])
            mask_t1_R_bins.append([])
            mask_t1_sum_S.append([])
            mask_t1_sum_R.append([])
            mask_t1_sum_K.append([])
            mask_t1_sum_M.append([])
            mask_s_sum.append([])
        # mask_t2_sum = []

        for i, input_tuple in enumerate(val_loader):
            # input and target
            batchsize = input_tuple[0].size(0)

            if args.gpus is None:
                input_data = input_tuple[0].cpu()

                target = input_tuple[-1].cpu()
            else:
                input_data = input_tuple[0].cuda(non_blocking=True)

                target = input_tuple[-1].cuda(non_blocking=True)

            # model forward function
            if 'dynstc' in args.arch:
                output, mask_stack_list, mask2_stack_list, gate_meta, dyn_outputs, soft_mask_stack_list = \
                    model(input=[input_data], tau=tau, is_training=False, curr_step=0, targets=target, epoch=epoch,
                          first_batch=i == 0)
            else:
                output, mask_stack_list, mask2_stack_list, gate_meta = \
                    model(input=[input_data], tau=tau, is_training=False, curr_step=0)

            # collect policy networks stats
            save_tap_dcp = False
            if save_tap_dcp:
                os.makedirs(os.path.join(exp_full_path, f'UNI_stats'), exist_ok=True)
                torch.save(mask_stack_list, os.path.join(exp_full_path, f'UNI_stats/tap_{i}.pth'))
            for num_layer in range(len(mask_t1_sum_S)):
                if args.pn_num_outputs == 1:
                    mask_t1_sum_S[num_layer].append(1 - torch.mean(mask_stack_list[num_layer][:, :, :, 0]).item())
                else:
                    mask_t1_sum_S[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 0]).item())
                if args.pn_num_outputs == 2:
                    mask_t1_sum_K[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 1]).item())
                elif args.pn_num_outputs == 1:
                    mask_t1_sum_K[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 0]).item())
                if args.pn_num_outputs > 2:
                    mask_t1_sum_R[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 1]).item())
                    mask_t1_sum_K[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 2]).item())
                if args.pn_num_outputs == 4:
                    mask_t1_sum_M[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 3]).item())

            # measure losses, accuracy and predictions
            upb_gflops_dynstc, real_gflops_dynstc = compute_gflops_dynstc(dyn_outputs, args.num_segments,
                                                                          args.batch_size, base_model_gflops)

            closs, rloss, bs_loss = my_criterion(output, target[:, 0], dyn_outputs['flops_real'],
                                                 dyn_outputs['flops_ori'][0], args.batch_size, args.den_target,
                                                 args.sparsity_lambda, soft_mask_stack_list, epoch, args)

            upb_batch_gflops_list.append(upb_gflops_dynstc)
            real_batch_gflops_list.append(real_gflops_dynstc)

            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            all_results.append(output)
            all_targets.append(target)

            dynstc_loss = closs.mean() + rloss.mean() + bs_loss.mean()
            closses.update(closs.mean().item(), batchsize)
            rlosses.update(rloss.mean().item(), batchsize)
            bslosses.update(bs_loss.mean().item(), batchsize)
            losses.update(dynstc_loss.item(), batchsize)

            top1.update(prec1.item(), batchsize)
            top5.update(prec5.item(), batchsize)

            # gather masks
            for layer_i, mask_stack in enumerate(mask_stack_list):
                mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)
                if args.pn_num_outputs == 1:
                    mask_stack_list_skipping[layer_i] += torch.sum(1 - mask_stack.detach(), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                'Loss{loss.val:.4f}({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'.
                                format(i, len(val_loader), batch_time=batch_time,
                                       loss=losses, top1=top1, top5=top5))

                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Cls', loss=closses)
                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Spar', loss=rlosses)
                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Bshp', loss=bslosses)
                print(print_output)

    upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
    real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # multi-label mAP

    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))
    print('==========================================================================================')
    print('policy networks stats:')
    for num_layer in range(len(mask_t1_sum_S)):
        if args.pn_num_outputs == 2 or args.pn_num_outputs == 1:
            print(
                f"{num_layer}: Temporal Skip - {np.around(np.mean(mask_t1_sum_S[num_layer]), decimals=3)}, Temporal Keep - {np.around(np.mean(mask_t1_sum_K[num_layer]), decimals=3)}")
        elif args.pn_num_outputs == 3:
            print(
                f"{num_layer}: Temporal Skip - {np.around(np.mean(mask_t1_sum_S[num_layer]), decimals=3)}, Temporal Reuse - {np.around(np.mean(mask_t1_sum_R[num_layer]), decimals=3)}, Temporal Keep - {np.around(np.mean(mask_t1_sum_K[num_layer]), decimals=3)}")
    print('==========================================================================================')

    usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
    print(usage_str)

    flops_per_clip = real_batch_gflops * args.num_segments

    return mAP, mmAP, top1.avg, top5.avg, usage_str, flops_per_clip


def set_random_seed(the_seed, args):
    np.random.seed(the_seed)
    torch.manual_seed(the_seed)


def compute_exp_decay_tau(epoch, init_tau, exp_decay_factor):
    return init_tau * np.exp(exp_decay_factor * epoch)


def get_current_temperature(num_epoch, exp_decay, init_tau, exp_decay_factor):
    if exp_decay:
        tau = compute_exp_decay_tau(num_epoch, init_tau, exp_decay_factor)
    else:
        tau = init_tau
    return tau


def get_policy_usage_str(upb_gflops, real_gflops):
    return "Equivalent GFLOPS: upb: %.4f   real: %.4f" % (upb_gflops.item(), real_gflops.item())


def get_recorders(number):
    return [Recorder() for _ in range(number)]


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]


def save_checkpoint(state, is_best, shall_backup, exp_full_path, decorator):
    if is_best:
        torch.save(state, '%s/models/%s.pth.tar' % (exp_full_path, decorator))
    if shall_backup:
        copyfile("%s/models/ckpt.best.pth.tar" % exp_full_path,
                 "%s/models/oldbest.%s.pth.tar" % (exp_full_path, decorator))


def setup_log_directory(exp_header, test_mode, args, logger):
    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    # if test_mode:
    #     exp_full_path = ospj(common.EXPS_PATH, args.test_from)
    # else:
    exp_full_path = ospj(common.EXPS_PATH, exp_full_name)

    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path, "models"))
    if not args.skip_log:
        logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path


def main(argv):
    t0 = time.time()
    inner_main(argv)
    print("Finished in %.4f seconds\n" % (time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1:])
