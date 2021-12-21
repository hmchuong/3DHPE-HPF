# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
from time import time
import torch
import numpy as np
import math
from .valid_angle_check import h36m_valid_angle_check_torch

def wing(x, omega=0.05, epsilon=0.02):
    dist = x.abs()
    delta_y1 = dist[dist < omega]
    delta_y2 = dist[dist >= omega]
    loss1 = omega * torch.log(1 + delta_y1 / epsilon)
    C = omega - omega * math.log(1 + omega / epsilon)
    loss2 = delta_y2 - C
    return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

def angle_orientation(predicted, omega=5e-3, epsilon=1e-3):
    
    # Relative position wrt Hip
    data = predicted.clone()
    data = data-data[:,:, 0, :]

    #1. Joint angle orientation constraint: Right Arm
    vtsr = data[:,:,14,:] - data[:,:,8,:] # RShoulder - Thorax
    vsrer = data[:,:,15,:] - data[:,:,14,:] # RElbow - RShoulder
    verwr = data[:,:,16,:] - data[:,:,15,:] # RWrist - RElbow
    
    #2. Joint angle orientation constraints: Left Arm
    vslel = data[:,:,12,:] - data[:,:,11,:] # LElbow - Lshoulder
    vtsl =   data[:,:,11,:] - data[:,:,8,:] # LShoulder - Thorax
    velwl =  data[:,:,13,:] - data[:,:,12,:] # LWrist - LElbow

    #3. Joint angle orientation constraints: Right Leg
    vhrkr = data[:,:,2,:] - data[:,:,1,:] # RKnee - RHip
    vphr = data[:,:,1,:] - data[:,:,0,:] #RHip - Hip
    vkrar = data[:,:,3,:] - data[:,:,8,:] #RFoot - RKnee

    #4. Joint angle orientation constraints: Right Leg
    vphl = data[:,:,4,:] - data[:,:,0,:] #LHip - Hip
    vhlkl = data[:,:,5,:] - data[:,:,4,:] #LKnee - LHip
    vklal = data[:,:,6,:] - data[:,:,5,:] #LFoot - LKnee

    limb_orientaion_loss = torch.clamp(-torch.cross(vtsr, vsrer, dim = -1) * verwr, min=0) + \
        torch.clamp(-torch.cross(vslel, vtsl, dim=-1) * velwl, min=0) + \
        torch.clamp(-torch.cross(vhrkr, vphr, dim=-1) * vkrar, min=0) + \
        torch.clamp(-torch.cross(vphl, vhlkl, dim=-1) * vklal, min=0)

    limb_orientaion_loss = limb_orientaion_loss.contiguous().view(-1).contiguous()
    if omega == -1 or epsilon == -1:
        limb_orientaion_loss = limb_orientaion_loss.mean()
    else:
        limb_orientaion_loss = wing(limb_orientaion_loss, omega=omega, epsilon=epsilon)

    #5. Joint angle orientation constraints: Torso
    vnh = data[:,:,10,:] - data[:,:,9,:] # Head - Neck
    vtp = data[:,:,0,:] - data[:,:,8,:] # Hip - Thorax
    
    torso_orientation_loss = torch.clamp(vnh*vtp, min=0) + torch.clamp(vtsr*vtsl, min=0) + torch.clamp(vphr*vphl, min=0) 
    torso_orientation_loss = torso_orientation_loss.contiguous().view(-1).contiguous()
    if omega == -1 or epsilon == -1:
        torso_orientation_loss = wing(torso_orientation_loss, omega=omega, epsilon=epsilon)
    else:
        torso_orientation_loss = torso_orientation_loss.mean()
    
    return limb_orientaion_loss, torso_orientation_loss

def smooth_l1(x):
    x = x.clone()
    #if x<1: x = 0.5x^2
    indices = torch.abs(x)<1
    x[indices] = 0.5*(x[indices]**2)

    #otherwise: x = |x|-0.5
    x[~indices] = torch.abs(x[~indices])-0.5
    return x

def ang_loss(predicted, target):
    def cs(p,j,c):
        u = p - j
        v = j - c
        uv= u*v #u.v
        Ak = torch.sum(uv,dim=-1) / (torch.norm(u, dim=-1)*torch.norm(v, dim=-1))
        indices = torch.isnan(Ak)
        Ak[indices] = 0
        return Ak
        
    # 1. Angle smoothness -- IN PROGRESS
    '''
    Joint angle loss merely constrains the angles of the limb joints:
        shoulders, elbows, hips and knees, i.e., M = 8. 
    '''
    def Acs(datad):
        # Relative position wrt Hip
        data = datad.clone()
        # import pdb; pdb.set_trace()
        data = data-data[:,:, 0, :]
        #J: Shoudlers, Elbows, Hips, Knees - L,R
        #P: Thorax, Shoulders, Hip, Hips - L,R
        #C: Elbows, Wrists, Knees, Foot - L,R
        j=[11, 12, 4, 5, 14, 15, 1, 2]
        p=[8, 11, 0, 4, 8, 14, 0, 1]
        c=[12, 13, 5, 6, 12, 16, 2, 3]
        return cs(data[:,:,p,:],data[:,:,j,:],data[:,:,c,:])
    
    # debug_time = time()
    A = Acs(predicted) #A shape [512,1,8]
    # print("cosine time", time() - debug_time)
    Ak = Acs(target)
    Langle = smooth_l1(torch.sum(A-Ak,dim=-1))
    Langle = torch.mean(Langle)
    
    return Langle



valid_ang = pickle.load(open('./data/h36m_valid_angle_0212.p', "rb"))
def ang_limit(y, y_gt, customized=False):
    ang_names = list(valid_ang.keys())
    y = y.reshape([-1, 17, 3])
    y_gt = y_gt.reshape([-1, 17, 3])
    ang_cos = h36m_valid_angle_check_torch(y)
    ang_cos_gt = h36m_valid_angle_check_torch(y_gt)
    loss = torch.tensor(0, dtype=y.dtype, device=y.device)
    N = 1
    smooth_l1 = torch.nn.SmoothL1Loss(reduction='sum')
    for an in ang_names:
        valid = torch.ones_like(ang_cos[an])
        lower_bound = valid_ang[an][0].min().item()
        if lower_bound >= -0.98:
            # loss += torch.exp(-b * (ang_cos[an] - lower_bound)).mean()
            if torch.any(ang_cos[an] < lower_bound):
                # loss += b * torch.exp(-(ang_cos[an][ang_cos[an] < lower_bound] - lower_bound)).mean()
                # lower_array = torch.ones_like(ang_cos[an][ang_cos[an] < lower_bound]) * lower_bound
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
                N += ang_cos[an][ang_cos[an] < lower_bound].shape[0]
                valid[ang_cos[an] < lower_bound] = 0
        upper_bound = valid_ang[an][1].max().item()
        if upper_bound <= 0.98:
            # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
            if torch.any(ang_cos[an] > upper_bound):
                # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                # upper_array = torch.ones_like(ang_cos[an][ang_cos[an] > upper_bound]) * upper_bound
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
                valid[ang_cos[an] > upper_bound] = 0
                N += ang_cos[an][ang_cos[an] > upper_bound].shape[0]
        if torch.any(valid > 0) and customized:
            loss += (ang_cos[an][valid > 0] - ang_cos_gt[an][valid > 0]).pow(2).sum()
            N += ang_cos[an][valid > 0].shape[0]
    loss = loss/N
    return loss

def mpjpe(predicted, target, top_k=1.0):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    dist = torch.norm(predicted - target, dim=len(target.shape)-1)
    if top_k == 1.0:
        return torch.mean(dist)
    else:
        dist = dist.contiguous().view(-1).contiguous()
        valid_loss, _ = torch.topk(dist, int(top_k * dist.size()[0]))
        return torch.mean(valid_loss)

def smooth_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # return torch.nn.SmoothL1Loss(reduction='mean')(predicted, target)
    return torch.mean(SmoothL1(torch.norm(predicted - target, dim=len(target.shape)-1, p=1)))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))