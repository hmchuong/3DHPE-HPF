# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.model_poseformer import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import *

from common.model_refinement import PoseRefinement
from common.metrics import compute_pdj


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# print(torch.cuda.device_count())


###################
args = parse_args()
# print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            if kps.shape[-1] > 2:
                kps[..., 2:] = normalize_screen_coordinates(kps[..., 2:], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = 100000
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

#########################################PoseTransformer

model_pos_train = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

model_pos = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
    
model_pos_refinement = PoseRefinement(fc_dim_in=5, num_fc=2, fc_dim=64, num_out=2)

################ load weight ########################
# posetrans_checkpoint = torch.load('./checkpoint/pretrained_posetrans.bin', map_location=lambda storage, loc: storage)
# posetrans_checkpoint = posetrans_checkpoint["model_pos"]
# model_pos_train = load_pretrained_weights(model_pos_train, posetrans_checkpoint)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos_refinement.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()
    model_pos_refinement = nn.DataParallel(model_pos_refinement)
    model_pos_refinement = model_pos_refinement.cuda()


if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)





test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


###################

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    # optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)
    optimizer = optim.AdamW(model_pos_refinement.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))


    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']
    
    best_score = 999999
    epoch = 0

    ## RESUME
    if True:
        chkpt_path = "checkpoint/refinement_fc_epoch-9_pdj-0.0128_improve-0.0019.pkl"
        checkpoint = torch.load(chkpt_path, map_location=lambda storage, loc: storage)
        if "model" in checkpoint:
            model_pos_refinement.load_state_dict(checkpoint["model"], strict=False)
        if "random_state" in checkpoint:
            train_generator.set_random_state(checkpoint['random_state'])
        if "lr" in checkpoint:
            lr = checkpoint['lr']
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "epoch" in checkpoint:
            epoch = checkpoint['epoch']
        if "best" in checkpoint:
            best_pdj = checkpoint['best']

    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    mse_loss = nn.MSELoss()

    # Pos model only
    while epoch < args.epochs:
    
        N = 0
        N_semi = 0
        # model_pos_train.train()
        model_pos_refinement.train()
        train_loss = 0
        epoch_ori_loss = 0
        batch_idx = 0
        for cameras_train, batch_3d, batch_2d in train_generator.next_epoch():
            break
            cameras_train = torch.from_numpy(cameras_train.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                cameras_train = cameras_train.cuda()
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            with torch.no_grad():
                predicted_3d_pos = model_pos_train(inputs_2d[:, :, :, 2:])
                # Save predicted_3d_pos + inputs_2d[40] => 512, 1, 17, 5 => 512, 17 * 5
                frame_idx = inputs_2d.shape[1] // 2
                pose_2d_gt = inputs_2d[:, frame_idx, ..., :2] # 512, 17, 2
                pose_2d = inputs_2d[:, frame_idx, ..., 2:] # 512, 17, 2
                pose_3d = predicted_3d_pos[:, 0] # 512, 17, 3
                # file_format = "data_refinement/epoch_{}_batch_{}.npz".format(epoch, batch_idx)
                # np.savez(file_format, pose_2d=pose_2d.cpu().numpy(), pose_2d_gt=pose_2d_gt.cpu().numpy(), pose_3d=pose_3d.cpu().numpy())
            
            inp_data = torch.cat([pose_2d, pose_3d], dim=2).permute(0, 2, 1)
            pose_2d_gt = pose_2d_gt.permute(0, 2, 1)
            pred = model_pos_refinement(inp_data)
            # loss = mse_loss(pred, pose_2d_gt)
            loss = mpjpe(pred.permute(0,2,1), pose_2d_gt.permute(0,2,1))
            ori_loss = mpjpe(pose_2d, pose_2d_gt.permute(0,2,1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pose_2d_gt.shape[0]
            epoch_ori_loss += ori_loss.item() * pose_2d_gt.shape[0]
            N += pose_2d_gt.shape[0]
            if batch_idx % 20 == 0:
                print("Train epoch {}/{} - batch {} - ori loss {:.4f} - loss {:.4f} - ori avg. loss {:.4f} - avg. loss {:.4f}".format(epoch + 1, args.epochs, batch_idx + 1, ori_loss.item(), loss.item(), epoch_ori_loss/N, train_loss/ N))
            batch_idx += 1
        
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()
            model_pos_refinement.eval()
            
            if not args.no_eval:
                eval_score = 0
                ori_score = 0
                batch_idx = 0
                N = 0
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    gt_2d = inputs_2d[..., :2]
                    inputs_2d = inputs_2d[..., 2:]

                    ##### apply test-time-augmentation (following Videopose3d)
                    
                    # prediction point
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    # gt point
                    gt_2d_flip = gt_2d.clone()
                    gt_2d_flip[:, :, :, 0] *= -1
                    gt_2d_flip[:, :, kps_left + kps_right, :] = gt_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

                    gt_2d, _ = eval_data_prepare(receptive_field, gt_2d, inputs_3d)

                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        gt_2d = gt_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                        inputs_3d = inputs_3d.cuda()
                    inputs_3d[:, :, 0] = 0
                    
                    predicted_3d_pos = model_pos(inputs_2d)
                    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                    
                    frame_idx = inputs_2d.shape[1] // 2
                    pose_2d = inputs_2d[:, frame_idx]
                    pose_3d = predicted_3d_pos[:, 0]
                    inp_data = torch.cat([pose_2d, pose_3d], dim=2).permute(0, 2, 1)

                    predicted_2d_pos = model_pos_refinement(inp_data)
                    predicted_2d_pos = predicted_2d_pos.permute(0, 2, 1).unsqueeze(1)

                    pose_2d = inputs_2d_flip[:, frame_idx]
                    pose_3d = predicted_3d_pos_flip[:, 0]
                    inp_data = torch.cat([pose_2d, pose_3d], dim=2).permute(0, 2, 1)

                    predicted_2d_pos_flip = model_pos_refinement(inp_data)
                    predicted_2d_pos_flip = predicted_2d_pos_flip.permute(0, 2, 1).unsqueeze(1)

                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                              joints_right + joints_left]

                    predicted_2d_pos_flip[:, :, :, 0] *= -1
                    predicted_2d_pos_flip[:, :, joints_left + joints_right] = predicted_2d_pos_flip[:, :,
                                                                              joints_right + joints_left]

                    predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                  keepdim=True)
                    
                    predicted_2d_pos = torch.mean(torch.cat((predicted_2d_pos, predicted_2d_pos_flip), dim=1), dim=1,
                                                  keepdim=True)
                    
                    
                    pdj = compute_pdj(predicted_2d_pos, gt_2d[:, frame_idx: frame_idx+1])
                    o_pdj = compute_pdj(inputs_2d[:, frame_idx], gt_2d[:, frame_idx: frame_idx+1])

                    # error = mpjpe(predicted_2d_pos, gt_2d[:, frame_idx: frame_idx+1])
                    # o_error = mpjpe(inputs_2d[:, frame_idx: frame_idx+1], gt_2d[:, frame_idx: frame_idx+1])
                    eval_score += pdj * predicted_2d_pos.shape[0]
                    ori_score += o_pdj * predicted_2d_pos.shape[0]
                    N += predicted_2d_pos.shape[0]
                    # import pdb; pdb.set_trace()
                    if batch_idx % 20 == 0:
                        print("Eval epoch {}/{} - batch {} - mpjpe {:.4f} - ori mpjpe {:.4f} - avg. mpjpe {:.4f} - avg. ori mpjpe {:.4f}".format(epoch + 1, args.epochs, batch_idx + 1, pdj, o_pdj, eval_score/ N, ori_score/ N))
                    batch_idx += 1
                
                mean_score = eval_score/ N
                mean_ori_score = ori_score/ N
                print("Eval epoch {}/{} - Mean MPJE: {:.4f} - Improvement: {:.4f}".format(epoch + 1, args.epochs, mean_score, mean_ori_score - mean_score))
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_chk_path = "checkpoint/refinement_conv1x1_epoch-{}_pdj-{:.4f}_improve-{:.4f}.pkl".format(epoch, mean_score, mean_ori_score - mean_score)
                    torch.save({
                        'epoch': epoch,
                        'lr': lr,
                        'best': best_score,
                        'random_state': train_generator.random_state(),
                        'optimizer': optimizer.state_dict(),
                        'model': model_pos_refinement.state_dict(),
                    }, best_chk_path)
                    print("Save model to", best_chk_path)
        
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_pdj = 0
    epoch_pdj_o = 0
    with torch.no_grad():
        model_pos.eval()
        model_pos_refinement.eval()
        N = 0
        N_2D = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            gt_2d = inputs_2d[..., :2]
            inputs_2d = inputs_2d[..., 2:]

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            # gt point
            gt_2d_flip = gt_2d.clone()
            gt_2d_flip[:, :, :, 0] *= -1
            gt_2d_flip[:, :, kps_left + kps_right, :] = gt_2d_flip[:, :, kps_right + kps_left, :]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

            gt_2d, _ = eval_data_prepare(receptive_field, gt_2d, inputs_3d)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                gt_2d = gt_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            
            # Predict 3D pose
            predicted_3d_pos = model_pos(inputs_2d)
            predicted_3d_pos_flip = model_pos(inputs_2d_flip)


            # Refine 2D pose
            frame_idx = inputs_2d.shape[1] // 2
            pose_2d = inputs_2d[:, frame_idx]
            pose_3d = predicted_3d_pos[:, 0]
            inp_data = torch.cat([pose_2d, pose_3d], dim=2).permute(0, 2, 1)

            predicted_2d_pos = model_pos_refinement(inp_data)
            predicted_2d_pos = predicted_2d_pos.permute(0, 2, 1).unsqueeze(1)

            pose_2d = inputs_2d_flip[:, frame_idx]
            pose_3d = predicted_3d_pos_flip[:, 0]
            inp_data = torch.cat([pose_2d, pose_3d], dim=2).permute(0, 2, 1)

            predicted_2d_pos_flip = model_pos_refinement(inp_data)
            predicted_2d_pos_flip = predicted_2d_pos_flip.permute(0, 2, 1).unsqueeze(1)

            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_2d_pos_flip[:, :, :, 0] *= -1
            predicted_2d_pos_flip[:, :, joints_left + joints_right] = predicted_2d_pos_flip[:, :,
                                                                        joints_right + joints_left]
            
            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)

            predicted_2d_pos = torch.mean(torch.cat((predicted_2d_pos, predicted_2d_pos_flip), dim=1), dim=1,
                                                  keepdim=True)

            # del inputs_2d, inputs_2d_flip
            torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

            # Compute for 2D poses
            # pdj = mpjpe(predicted_2d_pos, gt_2d[:, frame_idx: frame_idx+1]).item()
            pdj = compute_pdj(predicted_2d_pos, gt_2d[:, frame_idx: frame_idx+1]).item()
            o_pdj = compute_pdj(inputs_2d[:, frame_idx], gt_2d[:, frame_idx: frame_idx+1]).item()
            # o_pdj = mpjpe(inputs_2d[:, frame_idx: frame_idx+1], gt_2d[:, frame_idx: frame_idx+1]).item()

            epoch_pdj += pdj * predicted_2d_pos.shape[0]
            epoch_pdj_o += o_pdj * predicted_2d_pos.shape[0]
            N_2D +=  predicted_2d_pos.shape[0]


            # Compute MPJPE with new 2D pose
            # inputs_2d[:, frame_idx] = predicted_2d_pos[:, 0]
            # predicted_3d_pos = model_pos(inputs_2d)
            # inputs_2d_flip[:, frame_idx] = 
            # predicted_3d_pos_flip = model_pos(inputs_2d_flip)
            # predicted_3d_pos_flip[:, :, :, 0] *= -1
            # predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
            #                                                           joints_right + joints_left]


    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    pdj_2d = epoch_pdj / N_2D
    pdj_ori_2d = epoch_pdj_o / N_2D
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('PDJ original:', pdj_ori_2d)
    print('PDJ refinement:', pdj_2d)
    print('----------')

    return e1, e2, e3, ev, pdj_ori_2d, pdj_2d

if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    # if model_traj is not None and ground_truth is None:
    #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
    #     prediction += prediction_traj

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

else:
    chkpt_path = "checkpoint/refinement_fc_epoch-9_pdj-0.0128_improve-0.0019.pkl"
    checkpoint = torch.load(chkpt_path, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:
        model_pos_refinement.load_state_dict(checkpoint["model"])

    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))


    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        pdj_ori = []
        pdj_ref = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            e1, e2, e3, ev, po, pr = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            pdj_ori.append(po)
            pdj_ref.append(pr)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')
        print('PDJ            origin action-wise average:', round(np.mean(pdj_ori), 2))
        print('PDJ        refinement action-wise average:', round(np.mean(pdj_ref), 2))


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
