import torch

def compute_pdj(prediction, ground_truth):
    '''
    prediction: tensor
        N, 1, 17 or 16, 2
    groundtruth: tensor
        N, 1, 17 or 16, 2
    '''
    if len(prediction.shape) == 3:
        prediction = prediction.unsqueeze(1)
    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth.unsqueeze(1)

    # compute diagonal
    # xmin, _ = ground_truth[:, :, :, 0].min(2) # N, 1
    # xmax, _ = ground_truth[:, :, :, 0].max(2) # N, 1
    # ymin, _ = ground_truth[:, :, :, 1].min(2) # N, 1
    # ymax, _ = ground_truth[:, :, :, 1].max(2) # N, 1
    xmin = ground_truth[:, :, 1, 0]
    xmax = ground_truth[:, :, 11, 0]
    ymin = ground_truth[:, :, 1, 1]
    ymax = ground_truth[:, :, 11, 1]

    diagonals = torch.sqrt((xmax - xmin)**2 * 1.0 + (ymax - ymin)**2 * 1.0) # N, 1
    diagonals = diagonals.unsqueeze(-1)

    pred_xs = prediction[:, :, :, 0] # N, 1, 16 or 17
    pred_ys = prediction[:, :, :, 1] # N, 1, 16 or 17
    gt_ys = ground_truth[:, :, :, 1] # N, 1, 16 or 17
    gt_xs = ground_truth[:, :, :, 0] # N, 1, 16 or 17
    distance = torch.sqrt((gt_xs - pred_xs)**2 * 1.0 + (gt_ys - pred_ys)**2 * 1.0) # N, 1, 16 or 17

    # import pdb; pdb.set_trace();
    fraction = 0.2

    pdj = torch.sum(distance < fraction * diagonals, 2).squeeze() * 1.0 / prediction.shape[2]

    return pdj.mean()
