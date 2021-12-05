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

from ray import tune

torch.manual_seed(2021)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# print(torch.cuda.device_count())

def training_function(config):

    ###################
    args = parse_args()
    # args, unknown = parser.parse_known_args()
    # print(args)
    print(config)
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    print('Loading dataset...')
    dataset_path = '/home/ubuntu/PoseFormerPlus/data/data_3d_' + args.dataset + '.npz'
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
    keypoints = np.load('/home/ubuntu/PoseFormerPlus/data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
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
                if not action in dataset[subject]: continue
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

    ################ load weight ########################

    #################
    causal_shift = 0
    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
        model_pos_train = nn.DataParallel(model_pos_train)
        model_pos_train = model_pos_train.cuda()


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
        optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=args.lr_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, cooldown=1, verbose=True, min_lr=1e-8, factor=args.lr_decay)

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
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_generator.set_random_state(checkpoint['random_state'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']

        # Train generator set random state
        train_generator.set_random_state(np.random.RandomState(1234))

        print('** Note: reported losses are averaged over all frames.')
        print('** The final evaluation will be carried out after the last training epoch.')

        # Load hyper parameter from config
        use_smooth_L1 = config["smooth_l1"] # Using smooth L1 for MPJPE or not
        lambda1 = config["lambda1"] # mpjpe
        lambda2 = config["lambda2"] # bone length
        lambda3 = config["lambda3"] # angle orientation (limb)
        lambda4 = config["lambda4"] # angle orientation (torso)
        lambda5 = config["lambda5"] # joint angles of limbs
        lambda6 = config["lambda6"] # advanced angle constraints

        # Pos model only
        while epoch < args.epochs:
            start_time = time()
            epoch_loss_3d_train = 0
            epoch_loss_angle_train = 0
            epoch_loss_traj_train = 0
            epoch_loss_2d_train_unlabeled = 0
            N = 0
            N_semi = 0
            model_pos_train.train()
            batch_idx = 0
            debug_time = time()
            for cameras_train, batch_3d, batch_2d in train_generator.next_epoch():
                cameras_train = torch.from_numpy(cameras_train.astype('float32'))
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    cameras_train = cameras_train.cuda()
                inputs_traj = inputs_3d[:, :, :1].clone()
                inputs_3d[:, :, 0] = 0
                
                # print("Data loading time", time() - debug_time)
                debug_time = time()
                
                optimizer.zero_grad()

                # Predict 3D poses
                predicted_3d_pos = model_pos_train(inputs_2d)

                # torch.cuda.empty_cache()
                loss_3d_pos = smooth_mpjpe(predicted_3d_pos, inputs_3d) if use_smooth_L1 else mpjpe(predicted_3d_pos, inputs_3d)
                loss_bone_len = bone_len_constraint(predicted_3d_pos)
                loss_ang_orient_limb, loss_ang_orient_torso = angle_orientation_constraint(predicted_3d_pos)
                loss_limb_ang = limb_joint_angle(predicted_3d_pos, inputs_3d)
                loss_ang_constraint = advanced_angle_constraint(predicted_3d_pos, inputs_3d)

                # loss_3d_pos = torch.tensor(0).to(predicted_3d_pos.device)
                epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                epoch_loss_angle_train += inputs_3d.shape[0] * inputs_3d.shape[1] * 0
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                loss_total = lambda1 * loss_3d_pos + lambda2 * loss_bone_len \
                            + lambda3 * loss_ang_orient_limb + lambda4 * loss_ang_orient_torso \
                            + lambda5 * loss_limb_ang + lambda6 * loss_ang_constraint

                if batch_idx % 100 == 0:
                    print("Training: Epoch {} - Batch {}/{} - mpjpe loss: {:.4f} - bone loss: {:.4f} - ang orient loss: {:.4f} - limb ang loss: {:.4f} - ang const. loss: {:.4f} - total: {:.4f} - avg. mpjpe: {:.4f}".format(
                        epoch + 1, batch_idx + 1, train_generator.num_batches, loss_3d_pos.item(), lambda3 * loss_bone_len.item(), loss_ang_orient_limb.item() + lambda4 * loss_ang_orient_torso.item(), loss_limb_ang.item(), loss_ang_constraint.item(), loss_total.item(), epoch_loss_3d_train / N))

                
                loss_total.backward()
                # print("backward time", time() - debug_time)
                debug_time = time()

                optimizer.step()

                # print("optimization time", time() - debug_time)
                debug_time = time()

                # del inputs_3d, loss_3d_pos, predicted_3d_pos
                # torch.cuda.empty_cache()
                batch_idx += 1
                debug_time = time()

            losses_3d_train.append(epoch_loss_3d_train / N)
            torch.cuda.empty_cache()

            # End-of-epoch evaluation
            with torch.no_grad():
                model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
                model_pos.eval()

                epoch_loss_3d_valid = 0
                epoch_loss_traj_valid = 0
                epoch_loss_2d_valid = 0
                N = 0
                if not args.no_eval:
                    print("Evaluating ...")
                    print("Evaluating on test set:")
                    # Evaluate on test set
                    batch_idx = 0
                    for cam, batch, batch_2d in test_generator.next_epoch():
                        inputs_3d = torch.from_numpy(batch.astype('float32'))
                        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                        ##### apply test-time-augmentation (following Videopose3d)
                        inputs_2d_flip = inputs_2d.clone()
                        inputs_2d_flip[:, :, :, 0] *= -1
                        inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                        ##### convert size
                        inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
                        inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

                        if torch.cuda.is_available():
                            inputs_2d = inputs_2d.cuda()
                            inputs_2d_flip = inputs_2d_flip.cuda()
                            inputs_3d = inputs_3d.cuda()
                        inputs_3d[:, :, 0] = 0

                        predicted_3d_pos = model_pos(inputs_2d)
                        predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                        predicted_3d_pos_flip[:, :, :, 0] *= -1
                        predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                                joints_right + joints_left]

                        predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                    keepdim=True)

                        del inputs_2d, inputs_2d_flip

                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                        epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                        N += inputs_3d.shape[0] * inputs_3d.shape[1]

                        del inputs_3d, loss_3d_pos, predicted_3d_pos
                        if batch_idx % 20 == 0:
                            print("Evaluation: Epoch {} - Batch {}".format(
                                epoch + 1, batch_idx + 1))
                        batch_idx += 1
                        torch.cuda.empty_cache()

                    losses_3d_valid.append(epoch_loss_3d_valid / N)

                    # Evaluate on training set, this time in evaluation mode
                    epoch_loss_3d_train_eval = 0
                    epoch_loss_traj_train_eval = 0
                    epoch_loss_2d_train_labeled_eval = 0
                    N = 0
                    batch_idx = 0
                    print("Evaluating on train set:")
                    for cam, batch, batch_2d in train_generator_eval.next_epoch():
                        epoch_loss_3d_train_eval += 0
                        N += 1
                        break
                        if batch_2d.shape[1] == 0:
                            # This can only happen when downsampling the dataset
                            continue

                        inputs_3d = torch.from_numpy(batch.astype('float32'))
                        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                        inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)

                        if torch.cuda.is_available():
                            inputs_3d = inputs_3d.cuda()
                            inputs_2d = inputs_2d.cuda()

                        inputs_3d[:, :, 0] = 0

                        # Compute 3D poses
                        predicted_3d_pos = model_pos(inputs_2d)

                        del inputs_2d

                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                        epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                        N += inputs_3d.shape[0] * inputs_3d.shape[1]

                        del inputs_3d, loss_3d_pos, predicted_3d_pos
                        if batch_idx % 20 == 0:
                            print("Evaluation: Epoch {} - Batch {}".format(
                                epoch + 1, batch_idx + 1))
                        batch_idx += 1
                        torch.cuda.empty_cache()

                    losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                    # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                    epoch_loss_2d_train_unlabeled_eval = 0
                    N_semi = 0

            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
            else:

                log_str = '[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1] * 1000)
                print(log_str)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            lr = optimizer.param_groups[0]['lr']
            # scheduler.step(losses_3d_valid[-1])
            epoch += 1
            tune.report(mean_loss=losses_3d_valid[-1])


if __name__ == "__main__":
    def stopper(trial_id, result):
        if result["training_iteration"] > 1:
            return result["mean_loss"] * 1000 > 200 or result["mean_loss"] * 1000 < 66.5
        return False
    analysis = tune.run(
        training_function,
        metric="mean_loss",
        name="tuning_losses",
        mode='min',
        verbose=0,
        resources_per_trial={'gpu': 1, 'cpu': 20},
        config={
            "lambda1": tune.choice([0, 0.1, 0.5, 1.0]),
            "lambda2": tune.choice([0, 0.1, 0.5, 1.0]),
            "lambda3": tune.choice([0, 0.1, 0.5, 1.0]),
            "lambda4": tune.choice([0, 0.1, 0.5, 1.0]),
            "lambda5": tune.choice([0, 0.1, 0.5, 1.0]),
            "lambda6": tune.choice([0, 0.1, 0.5, 1.0]),
            "smooth_l1": tune.choice([True, False])
        }, num_samples=30, stop=stopper)

    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    print(df)
