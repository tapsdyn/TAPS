# TAPS - Temporal Attention-based Pruning and Scaling

Official implementation of "**TAPS: Temporal Attention-based Pruning and Scaling for Efficient Video Action Recognition**"

![Arch_tnr](https://github.com/tapsdyn/TAPS/assets/141319872/e386cddd-e87d-4ac3-86a8-22692da4e2ae)


## Dependencies
To install requirements:
```
pip install -r requirements.txt
```
This repository has been tested in Ubuntu 18.04 and Cuda 11.3.1

## Datasets Preparations
Download the datasets ([SomethingV2](https://developer.qualcomm.com/software/ai-datasets/something-something), [Jester](https://developer.qualcomm.com/software/ai-datasets/jester), [ActivityNetv1.3](http://activity-net.org/) from the official websites)<br />
Mini-Kinetics is downloaded from [https://github.com/cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset)<br />

Note: to extract jpg frames data from ActivityNetv1.3 and Mini-Kinetics videos, we followed the instructions in FrameExit repository:<br />
[https://github.com/Qualcomm-AI-research/FrameExit/tree/main](https://github.com/Qualcomm-AI-research/FrameExit/tree/main)<br />

In `common.py`,  modify `ROOT_DIR`, `DATA_PATH` and `EXPS_PATH` to setup the dataset path and logs path (where to save checkpoints and logs)

The datasets file arrangement should match the relative paths as appear in the train/val splits text files of each dataset under the `data` folder.

## Inference
In order to reproduce the results for the ResNet-50 model on `Mini-Kinetics` in online mode:

1. Download the pretrained model from [our drive](https://drive.google.com/drive/folders/12KYVL9y_c9jJcvprx3aMnMiniljCm7qv?usp=sharing) and place it in a folder names "taps_chekpoints" under your experiment directory `EXPS_PATH`
2. Run the following command (this assumes 2 GPUs)
```bash
python main.py minik_aug23 RGB --arch res50_dynstc_net --num_segments 8 --npb --gate_hidden_dim 1024 --gbn --grelu --shift --online_shift --TDCP_attn --TDCP_soft_mask --TDCP_attn_PE --TDCP_mul_MHA --TDCP_attn_num_heads 4 --TDCP_mul_MHA_disable_ffn --TDCP_attn_online_inf --pn_num_output 1 --batch-size 32 -j 10 --gpus 0 1 --test_from taps_checkpoints/minik_online_8_frames/ckpt.best.pth.tar --skip_log
```
In order to reproduce the results for the ResNet-50 model on `ActivityNet_v1.3` in offline mode:

1. Download the pretrained model from [our drive](https://drive.google.com/drive/folders/12KYVL9y_c9jJcvprx3aMnMiniljCm7qv?usp=sharing) and place it in a folder names "taps_chekpoints" under your experiment directory `EXPS_PATH`
2. Run the following command (this assumes 2 GPUs)
```bash
python main.py activitynet_1_3_aug23 RGB --arch res50_dynstc_net --num_segments 8 --npb --gate_hidden_dim 1024 --gbn --grelu --TDCP_attn --TDCP_soft_mask --TDCP_attn_PE --TDCP_mul_MHA --TDCP_attn_num_heads 4 --TDCP_mul_MHA_disable_ffn --pn_num_output 1 --batch-size 32 -j 7 --gpus 0 1 --test_from taps_checkpoints/activitynet_offline_8_frames/ckpt.best.pth.tar --skip_log
```


## Training
To train use the following command (this is for Mini-Kinetics, 8 frames, online mode). This assumes 4 GPUs for batch size of 64.
Note to set the "exp_header" argument to your desired experiment name.
```bash
python main.py minik_aug23 RGB --arch res50_dynstc_net --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 1e-4 --npb --init_tau 0.67 --gate_hidden_dim 1024 --gbn --grelu --shift --online_shift --sparsity_lambda 5.0 --den_target 0.45 --TDCP_attn --TDCP_soft_mask --TDCP_attn_PE --TDCP_mul_MHA --TDCP_attn_num_heads 4 --TDCP_mul_MHA_disable_ffn --pn_num_output 1 --sparsity_warmup_start 10 --sparsity_warmup_end 30 --warmup_hard_gates_disable --batch-size 64 -j 20 --gpus 0 1 2 3 --exp_header minik_r50_bs64_taps_onl_dt045
```

## Results
