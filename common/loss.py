# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import torch
import numpy as np
from .valid_angle_check import h36m_valid_angle_check_torch

valid_ang = pickle.load(open('./data/h36m_valid_angle.p', "rb"))
def angle_loss(y, y_gt):
    ang_names = list(valid_ang.keys())
    y = y.reshape([-1, 17, 3])
    y_gt = y_gt.reshape([-1, 17, 3])
    ang_cos = h36m_valid_angle_check_torch(y)
    ang_cos_gt = h36m_valid_angle_check_torch(y_gt)
    loss = torch.tensor(0, dtype=y.dtype, device=y.device)
    b = 1
    for an in ang_names:
        valid = torch.ones_like(ang_cos[an])
        if an != "Spine2HipPlane":
            lower_bound = valid_ang[an][0]
            if lower_bound >= -0.98:
                # loss += torch.exp(-b * (ang_cos[an] - lower_bound)).mean()
                if torch.any(ang_cos[an] < lower_bound):
                    # loss += b * torch.exp(-(ang_cos[an][ang_cos[an] < lower_bound] - lower_bound)).mean()
                    loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
                    valid[ang_cos[an] < lower_bound] = 0
            upper_bound = valid_ang[an][1]
            if upper_bound <= 0.98:
                # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
                if torch.any(ang_cos[an] > upper_bound):
                    # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                    loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
                    valid[ang_cos[an] > upper_bound] = 0
        if not an in ["Spine2HipPlane1", "Spine2HipPlane2"]:
            if torch.any(valid > 0):
                loss += (ang_cos[an][valid > 0] - ang_cos_gt[an][valid > 0]).pow(2).mean()
    return loss


def angle_loses(predicted, target):
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
    H36M_NAMES[0]  = 'Hip'
    H36M_NAMES[1]  = 'RHip'
    H36M_NAMES[2]  = 'RKnee'
    H36M_NAMES[3]  = 'RFoot'
    H36M_NAMES[6]  = 'LHip'
    H36M_NAMES[7]  = 'LKnee'
    H36M_NAMES[8]  = 'LFoot'
    H36M_NAMES[12] = 'Spine'
    H36M_NAMES[13] = 'Thorax'
    H36M_NAMES[14] = 'Neck/Nose'
    H36M_NAMES[15] = 'Head'
    H36M_NAMES[17] = 'LShoulder'
    H36M_NAMES[18] = 'LElbow'
    H36M_NAMES[19] = 'LWrist'
    H36M_NAMES[25] = 'RShoulder'
    H36M_NAMES[26] = 'RElbow'
    H36M_NAMES[27] = 'RWrist'
    '''

    PAPER1 = False
    PAPER2 = True
    L=0

    #------------------------------------------------------------------------------------------------------------
    ''' ---- IN PROGRESS
    #Paper1: Multi-scale Recalibration with Advanced Geometry Constraints for 3D Human Pose Estimation
    if(PAPER1):  
        #Advanced Geometric Constraints:
        #1. Symmetrical  bone  length ratio constraint
        Llen1 = 0
        #2. Symmetrical  bone  length  constraint
        Llen2=0
        for i in range(0,N):
            Llen2 += mod(Si) * 
        #3. Joint angle orientation constraint: Lower Arm
        LarmR = max(np.dot(np.cross(vtsr,vsrer), verwr) ,0)
        #4. Joint angle orientation constraints: Torso
        Langle1 = max(np.dot(np.cross(vtsr,vsrer), verwr) ,0) + \
            max(np.dot(np.cross(vslel,vtsl), velwl) ,0) + \
            max(np.dot(np.cross(vhrkr,vsrer), vkrar) ,0) + \
            max(np.dot(np.cross(vphl,vhlkl), vkrar) ,0)
        #5. Joint angle orientation constraints: Torso
        Langle2 = max(np.dot(vnh,vtp),0) + max(np.dot(vtsr,vtsl),0) + max(np.dot(vphr,vpsl),0) 
        #Total geometric contraint:
        #Lgeo = Ldep 
        L+=Lgeo
    '''


    #-----------------------------------------------------------------------------------------------------------
    #Paper2: A Joint Relationship A ware Neural Network for Single-Image 3D Human Pose Estimation 
    if(PAPER2):
        # Section C. The Local Joint Relationship Awareness 

        #Given Functions as in Paper:
        """
        SmoothL1 Smoothness function 
        input: tensor
        output: tensor
        """
        def SmoothL1(x):
            #if x<1: x = 0.5x^2
            indices = torch.abs(x)<1
            x[indices] = 0.5*x[indices]**2

            #otherwise: x = |x|-0.5
            x[~indices] = torch.abs(x[~indices])-0.5
            return x

        """
        cs Cosine similarity measure
        params: parent joint, current joint, child joint
        output: tensor
        """
        def cs(p,j,c):
            Ak = torch.dot(p - j , j - c) / torch.norm(p - j, dim=len(target.shape)-1)*torch.norm(j - c, dim=len(target.shape)-1)
            return Ak

        # 1. Angle smoothness -- IN PROGRESS
        '''
        Joint angle loss merely constrains the angles of the limb joints:
            shoulders, elbows, hips and knees, i.e., M = 8. 
        '''
        '''
        sum = 0
        for i in [1,2,3]:
            xp = predicted[:,:,i,0].reshape(-1)
            yp = predicted[:,:,i,1].reshape(-1)
            zp = predicted[:,:,i,2].reshape(-1)
            xj = predicted[:,:,i,0].reshape(-1)
            yj = predicted[:,:,i,1].reshape(-1)
            zj = predicted[:,:,i,2].reshape(-1)
            xc = predicted[:,:,i,0].reshape(-1)
            yc = predicted[:,:,i,1].reshape(-1)
            zc = predicted[:,:,i,2].reshape(-1)
            sum += cs([xp,yp,zp],[xj,yj,zj],[xc, yc, zc])
        Langle = SmoothL1(sum)
        L += Langle
        '''

        # 2. Joint smoothness --- DONE
        #predicted = torch.from_numpy(predicted.astype('float32'))
        Ljoint = SmoothL1(torch.norm(predicted - target, dim=len(target.shape)-1))
        Ljoint = torch.mean(Ljoint)
        L += Ljoint

    return L

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
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
        
        """
        cs Cosine similarity measure
        params: parent joint, current joint, child joint
        output: tensor
        """
        def cs(p,j,c):
            Ak = torch.dot(p - j , j - c) / torch.norm(p - j, dim=len(target.shape)-1)*torch.norm(j - c, dim=len(target.shape)-1)
            return Ak

        # 1. Angle smoothness -- IN PROGRESS
        '''
        sum = 0
        #predicted1 = predicted.cpu().numpy()
        #predicted2 = predicted1.detach().numpy()
        for i in [1,2,3]:
            xp = predicted[:,:,i,0].reshape(-1)
            yp = predicted[:,:,i,1].reshape(-1)
            zp = predicted[:,:,i,2].reshape(-1)

            xj = predicted[:,:,i,0].reshape(-1)
            yj = predicted[:,:,i,1].reshape(-1)
            zj = predicted[:,:,i,2].reshape(-1)

            xc = predicted[:,:,i,0].reshape(-1)
            yc = predicted[:,:,i,1].reshape(-1)
            zc = predicted[:,:,i,2].reshape(-1)
            sum += cs([xp,yp,zp],[xj,yj,zj],[xc, yc, zc])
        Langle = SmoothL1(sum)
        L += Langle
        '''
        
        # 2. Joint smoothness --- DONE
        #predicted = torch.from_numpy(predicted.astype('float32'))
        Ljoint = SmoothL1(torch.norm(predicted - target, dim=len(target.shape)-1))
        Ljoint = torch.mean(Ljoint)
        L += Ljoint

    return L

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
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