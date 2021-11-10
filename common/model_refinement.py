from torch import nn
from torch.nn import functional as F

# class PoseRefinement(nn.Module):
#     def __init__(self, fc_dim_in, num_fc, fc_dim, num_out):
#         super().__init__()
#         self.fc_layers = []
#         for k in range(num_fc):
#             fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
#             self.add_module("fc{}".format(k + 1), fc)
#             self.fc_layers.append(fc)
#             fc_dim_in = fc_dim
        
#         self.predictor = nn.Conv1d(fc_dim_in, num_out, kernel_size=1, stride=1, padding=0)
    
#     def forward(self, x):
#         for layer in self.fc_layers:
#             x = F.relu(layer(x))
#         return self.predictor(x)

class PoseRefinement(nn.Module):
    def __init__(self, fc_dim_in, num_fc, fc_dim, num_out):
        super().__init__()
        n_joints = 17
        fc_dim_in = fc_dim_in * n_joints
        out_channels = num_out * n_joints
        self.fc_layers = []
        for k in range(num_fc):
            # fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            fc = nn.Linear(fc_dim_in, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
        
        self.predictor = nn.Linear(fc_dim_in, out_channels)
    
    def forward(self, x):
        N, C, M = x.shape
        x = x.contiguous().view(N, C * M)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        x = self.predictor(x)
        x = x.contiguous().view(N, 2, M)
        return x