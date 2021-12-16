# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
from time import time
import torch
import math
import numpy as np
from .valid_angle_check import h36m_valid_angle_check_torch

def bone_len_constraint(predicted):

    # Relative position wrt Hip
    data = predicted.clone()
    data = data-data[:,:, 0, :]

    L = 0
    # Shoulder
    rshoulder = torch.norm(data[:,:,14,:] - data[:,:,8,:], dim=-1)
    lshoulder = torch.norm(data[:,:,11,:] - data[:,:,8,:], dim=-1)
    L += torch.nn.functional.mse_loss(rshoulder, lshoulder, reduction='mean')

    # Above arm
    raarm = torch.norm(data[:,:,15,:] - data[:,:,14,:], dim=-1)
    laarm = torch.norm(data[:,:,12,:] - data[:,:,11,:], dim=-1)
    L += torch.nn.functional.mse_loss(raarm, laarm, reduction='mean')

    # Below arm
    rbarm = torch.norm(data[:,:,16,:] - data[:,:,15,:], dim=-1)
    lbarm = torch.norm(data[:,:,13,:] - data[:,:,12,:], dim=-1)
    L += torch.nn.functional.mse_loss(rbarm, lbarm, reduction='mean')

    # Hip
    rhip = torch.norm(data[:, :, 1, :] - data[:, :, 0, :], dim=-1)
    lhip = torch.norm(data[:, :, 4, :] - data[:, :, 0, :], dim=-1)
    L += torch.nn.functional.mse_loss(rhip, lhip, reduction='mean')

    # Thigh
    rthigh = torch.norm(data[:, :, 2, :] - data[:, :, 1, :], dim=-1)
    lthigh = torch.norm(data[:, :, 5, :] - data[:, :, 4, :], dim=-1)
    L += torch.nn.functional.mse_loss(rthigh, lthigh, reduction='mean')

    # Leg
    rleg = torch.norm(data[:, :, 2, :] - data[:, :, 3, :], dim=-1)
    lleg = torch.norm(data[:, :, 5, :] - data[:, :, 6, :], dim=-1)
    L += torch.nn.functional.mse_loss(rleg, lleg, reduction='mean')

    return L


def angle_orientation_constraint(predicted, top_k=1.0):
    
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

    # print("limb",limb_orientaion_loss.mean())
    
    # limb_orientaion_loss, ind = (
    #         limb_orientaion_loss.contiguous()
    #         .view(
    #             -1,
    #         )
    #         .contiguous()
    #         .sort()
    #     )
    # min_value = limb_orientaion_loss[min(100000, limb_orientaion_loss.numel() - 1)]
    # threshold = max(min_value, 0.05)

    # limb_orientaion_loss = limb_orientaion_loss[limb_orientaion_loss < threshold]
    limb_orientaion_loss = limb_orientaion_loss.contiguous().view(-1).contiguous()
    limb_orientaion_loss, idxs = torch.topk(limb_orientaion_loss, int(top_k * limb_orientaion_loss.size()[0]))
    limb_orientaion_loss = torch.mean(limb_orientaion_loss)
    # print("limb mean", limb_orientaion_loss)

    #5. Joint angle orientation constraints: Torso
    vnh = data[:,:,10,:] - data[:,:,9,:] # Head - Neck
    vtp = data[:,:,0,:] - data[:,:,8,:] # Hip - Thorax
    
    torso_orientation_loss = torch.clamp(vnh*vtp, min=0) + torch.clamp(vtsr*vtsl, min=0) + torch.clamp(vphr*vphl, min=0) 
    # print("torso",torso_orientation_loss.shape,limb_orientaion_loss.max(), limb_orientaion_loss.min())
    # torso_orientation_loss, ind = (
    #         torso_orientation_loss.contiguous()
    #         .view(
    #             -1,
    #         )
    #         .contiguous()
    #         .sort()
    #     )
    # min_value = torso_orientation_loss[min(100000, limb_orientaion_loss.numel() - 1)]
    # threshold = max(min_value, 0.05)
    
    # torso_orientation_loss = torso_orientation_loss[torso_orientation_loss < threshold]
    torso_orientation_loss = torso_orientation_loss.contiguous().view(-1).contiguous()
    torso_orientation_loss, idxs = torch.topk(torso_orientation_loss, int(top_k * torso_orientation_loss.size()[0]))
    torso_orientation_loss = torch.mean(torso_orientation_loss)
    
    return limb_orientaion_loss, torso_orientation_loss

def limb_joint_angle(predicted, target):
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
        # j = [1, 2, 4, 5, 7, 7, 7, 8, 9, 11, 12, 14, 15]
        # p = [0, 1, 0, 4, 0, 0, 0, 7, 8, 8, 11, 8, 14]
        # c = [2, 3, 5, 6, 8, 11, 14, 9, 10, 12, 13, 15, 16]
        return cs(data[:,:,p,:],data[:,:,j,:],data[:,:,c,:])
    
    # debug_time = time()
    A = Acs(predicted) #A shape [512,1,8]
    # print("cosine time", time() - debug_time)
    Ak = Acs(target)
    Langle = SmoothL1(torch.sum(A-Ak,dim=-1))
    Langle = torch.mean(Langle)
    
    return Langle

def angle_losses(predicted, target):
    """
    Modified joint position error. 
    Use this function in run_poseformer.py 
        after Line 311: loss_ang = angle_loss(predicted_3d_pos, inputs_3d)
    """

    assert predicted.shape == target.shape

    #print("Predicted shape: ", predicted.shape) #OP: torch.Size([512, 1, 17, 3]) 
    # Joint space configuration on File: hm36m_dataset.py, Line 247

    '''
    These are XYZ coordinates relative to the pelvis-joint (X00 = 0, Y00 = 0, Z00 = 0 is the pelvis-joint), 
        expressed in millimeters (mm).
    H36M_NAMES[0]  = 'Hip'      0
    H36M_NAMES[1]  = 'RHip'     1
    H36M_NAMES[2]  = 'RKnee'    2
    H36M_NAMES[3]  = 'RFoot'    3
    H36M_NAMES[6]  = 'LHip'     4
    H36M_NAMES[7]  = 'LKnee'    5
    H36M_NAMES[8]  = 'LFoot'    6
    H36M_NAMES[12] = 'Spine'    7
    H36M_NAMES[13] = 'Thorax'   8
    H36M_NAMES[14] = 'Neck/Nose'9
    H36M_NAMES[15] = 'Head'     10
    H36M_NAMES[17] = 'LShoulder'11
    H36M_NAMES[18] = 'LElbow'   12
    H36M_NAMES[19] = 'LWrist'   13
    H36M_NAMES[25] = 'RShoulder'14
    H36M_NAMES[26] = 'RElbow'   15  
    H36M_NAMES[27] = 'RWrist'   16
    '''

    PAPER1 = False
    PAPER2 = True
    L=0

    #------------------------------------------------------------------------------------------------------------
    # ---- IN PROGRESS
    #Paper1: Multi-scale Recalibration with Advanced Geometry Constraints for 3D Human Pose Estimation
    if(PAPER1):  
        '''
        #Advanced Geometric Constraints:
        #1. Symmetrical  bone  length ratio constraint
            #Ri is a set of bones which follow a fixed ratio,
            #ri is the  average  ratio  of  bone  length
        
        #4 groups of bones:
        #Rarm = {left/right lower/upper arms}, 
        #Rleg = { left/right lower/upper legs},
        #Rshoulder = { left/right shoulder bones}, 
        #Rhip = {left/right hip bones}
        
        Llen1 = 0
        Ri = {}
        #Ratio of lengths  = []
        r=[]
        #l length of bone
        #le canonical skeleton bone length
        for i in Ri:
            for e in e[Ri]:
                Llen1 += (l[e]/l_[e] - r[i])**2
    
        #2. Symmetrical  bone  length  constraint
        Llen2=0
        for i in range(0,N):
            Llen2 += abs(Si) * 
        
        '''
        debug_time = time()
        # Relative position wrt Hip
        data = predicted
        data = data-data[:,:, 0, :]

        #3. Joint angle orientation constraint: Lower Arm
        vtsr = data[:,:,14,:] - data[:,:,8,:] #RShoulder - Thorax
        vsrer = data[:,:,15,:] - data[:,:,14,:] #RElbow - RShoulder
        verwr = data[:,:,16,:] - data[:,:,15,:] #RWrist - RElbow
        LarmR = torch.clamp(torch.cross(vtsr,vsrer,dim=-1) * verwr, min=0)
        LarmR = torch.mean(LarmR)
        #print("LarmR: ",LarmR)
        L+=LarmR
        
        #4. Joint angle orientation constraints: Torso
        vslel = data[:,:,12,:] - data[:,:,11,:]#LElbow - Lshoulder
        vtsl =   data[:,:,11,:] - data[:,:,8,:] #LShoulder - Thorax
        velwl =  data[:,:,13,:] - data[:,:,12,:] #LWrist - LElbow
        vhrkr = data[:,:,2,:] - data[:,:,1,:] #RKnee - RHip
        vkrar = data[:,:,14,:] - data[:,:,8,:] #RAnkle - RKnee
        vphl = data[:,:,4,:] - data[:,:,0,:] #LHip - Hip
        vhlkl = data[:,:,5,:] - data[:,:,4,:] #LKnee - LHip
        Langle1 = torch.clamp(torch.cross(vtsr,vsrer) * verwr, min=0) + \
            torch.clamp(torch.cross(vslel,vtsl)* velwl, min=0) + \
            torch.clamp(torch.cross(vhrkr,vsrer)* vkrar, min=0) + \
            torch.clamp(torch.cross(vphl,vhlkl)* vkrar, min=0)
        Langle1 = torch.mean(Langle1)
        #print("Langle1: ",Langle1)
        L+=Langle1

        #5. Joint angle orientation constraints: Torso
        vnh = data[:,:,10,:] - data[:,:,9,:] #Head - Neck
        vtp = data[:,:,0,:] - data[:,:,8,:] #Hip - Thorax
        vphr = data[:,:,1,:] - data[:,:,0,:] #RHip - Hip
        Langle2 = torch.clamp(vnh*vtp, min=0) + torch.clamp(vtsr*vtsl, min=0) + torch.clamp(vphr*vphl, min=0) 
        Langle2 = torch.mean(Langle2)
        #print("Langle2: ",Langle2)
        L+=Langle2
        # print("Angle loss 1", time() - debug_time)

    #-----------------------------------------------------------------------------------------------------------
    #Paper2: A Joint Relationship A ware Neural Network for Single-Image 3D Human Pose Estimation 
    if(PAPER2):
        debug_time = time()
        # Section C. The Local Joint Relationship Awareness 

        #Given Functions as in Paper:
        """
        SmoothL1 Smoothness function 
        input: tensor
        output: tensor
        """

        """
        cs Cosine similarity measure
        params: parent joint, current joint, child joint
        output: tensor
        """
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
            # j = [1, 2, 4, 5, 7, 7, 7, 8, 9, 11, 12, 14, 15]
            # p = [0, 1, 0, 4, 0, 0, 0, 7, 8, 8, 11, 8, 14]
            # c = [2, 3, 5, 6, 8, 11, 14, 9, 10, 12, 13, 15, 16]
            return cs(data[:,:,p,:],data[:,:,j,:],data[:,:,c,:])
        
        # debug_time = time()
        A = Acs(predicted) #A shape [512,1,8]
        # print("cosine time", time() - debug_time)
        Ak = Acs(target)
        Langle = SmoothL1(torch.sum(A-Ak,dim=-1))
        Langle = torch.mean(Langle)
        # print("Langle: ", Langle)
        L += Langle
        

        # 2. Joint smoothness --- DONE
        #predicted = torch.from_numpy(predicted.astype('float32'))
        # Ljoint = SmoothL1(torch.norm(predicted - target, dim=len(target.shape)-1))
        # Ljoint = torch.mean(Ljoint)
        # print("Ljoint: ",Ljoint)
        # L += Ljoint
        # print("Angle loss 2", time() - debug_time)
    return L

def SmoothL1(x):
    x = x.clone()
    #if x<1: x = 0.5x^2
    indices = torch.abs(x)<1
    x[indices] = 0.5*(x[indices]**2)

    #otherwise: x = |x|-0.5
    x[~indices] = torch.abs(x[~indices])-0.5
    return x

valid_ang = pickle.load(open('./data/h36m_valid_angle_0212.p', "rb"))
def advanced_angle_constraint(y, y_gt, customized=False):
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
                lower_array = torch.ones_like(ang_cos[an][ang_cos[an] < lower_bound]) * lower_bound
                loss += smooth_l1(ang_cos[an][ang_cos[an] < lower_array], lower_array)
                N += ang_cos[an][ang_cos[an] < lower_bound].shape[0]
                valid[ang_cos[an] < lower_bound] = 0
        upper_bound = valid_ang[an][1].max().item()
        if upper_bound <= 0.98:
            # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
            if torch.any(ang_cos[an] > upper_bound):
                # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                upper_array = torch.ones_like(ang_cos[an][ang_cos[an] > upper_bound]) * upper_bound
                loss += smooth_l1(ang_cos[an][ang_cos[an] > upper_bound], upper_array)
                valid[ang_cos[an] > upper_bound] = 0
                N += ang_cos[an][ang_cos[an] > upper_bound].shape[0]
        if torch.any(valid > 0) and customized:
            loss += (ang_cos[an][valid > 0] - ang_cos_gt[an][valid > 0]).pow(2).sum()
            N += ang_cos[an][valid > 0].shape[0]
    loss = loss/N
    return loss
    
def wing_loss(pred, target, omega=0.01, epsilon=2):
    y = target
    y_hat = pred
    delta_y = (y - y_hat).abs()
    delta_y1 = delta_y[delta_y < omega]
    delta_y2 = delta_y[delta_y >= omega]
    loss1 = omega * torch.log(1 + delta_y1 / epsilon)
    C = omega - omega * math.log(1 + omega / epsilon)
    loss2 = delta_y2 - C
    return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

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
        valid_loss, idxs = torch.topk(dist, int(top_k * dist.size()[0]))
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