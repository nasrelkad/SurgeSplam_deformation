# utils/deform_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDeformMLP(nn.Module):
    """
    Endo-4DGS style: take canonical gaussian info + time → deltas.
    This is intentionally small so we can run it per frame.
    """
    def __init__(self, in_dim=3+1, hidden=64):
        super().__init__()
        # in_dim: (xyz_canon=3) + (t=1). You can add view, depth, mask later.
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.input_dim = in_dim
        self.hidden_dim = hidden
        # outputs:
        #  - dxyz (3)
        #  - dlog_scale (3)
        #  - drot_aa (3)  axis-angle delta
        #  - dopacity (1)
        self.fc_out = nn.Linear(hidden, 3 + 3 + 3 + 1)

    def forward(self, xyz_canon, t_scalar):
        # xyz_canon: [G,3]
        # t_scalar: float or tensor []
        if not torch.is_tensor(t_scalar):
            t = torch.tensor([t_scalar], device=xyz_canon.device, dtype=xyz_canon.dtype)
        else:
            t = t_scalar.to(xyz_canon.device, xyz_canon.dtype)
        t = t.view(1, 1).expand(xyz_canon.shape[0], 1)   # [G,1]

        x = torch.cat([xyz_canon, t], dim=-1)           # [G,4]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc_out(h)

        # Apply tanh to bound outputs and prevent extreme values
        dxyz       = torch.tanh(out[..., 0:3]) * 0.05      # Max ±0.05 position delta
        dlogscale  = torch.tanh(out[..., 3:6]) * 0.5       # Max ±0.5 log-scale delta
        drot_aa    = torch.tanh(out[..., 6:9]) * 0.3       # Max ±0.3 radians (~17°)
        dopacity   = torch.tanh(out[..., 9:10]) * 2.0      # Max ±2.0 logit delta
        return dxyz, dlogscale, drot_aa, dopacity
