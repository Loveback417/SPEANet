import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import RotatedShared2FCBBoxHead
from mmrotate.registry import MODELS


class FAA(nn.Module):
    """
    faa (Frequency Alignment Aggregation) Module.
    Input: [B, 1, 7, 7] → Output: (rot_inv_mag, θe, x_aligned)
        rot_inv_mag: Rotation-invariant magnitude vector [B, m]
        θe: Estimated principal angle [B,]
        x_aligned: Spatial image aligned to principal direction [B, 1, 7, 7] (optional)
    """

    def __init__(self, eps=1e-8, return_aligned=True):
        super(FAA, self).__init__()
        self.eps = eps
        self.return_aligned = return_aligned  # Whether to return aligned image

        H, W = 7, 7
        h_idx = torch.arange(H)
        w_idx = torch.arange(W // 2 + 1)

        h_shift = torch.fft.fftshift(h_idx - H // 2, dim=0)  # [-3,-2,-1,0,1,2,3]
        w_shift = torch.cat([w_idx[:W // 2], torch.tensor([-W // 2])])  # [0,1,2,-3]

        y, x_grid = torch.meshgrid(h_shift, w_shift)
        rho = torch.sqrt(x_grid ** 2 + y ** 2)
        theta = torch.atan2(y, x_grid)
        theta = (theta + 2 * math.pi) % (2 * math.pi)  # [0, 2π)

        mask = rho > self.eps
        valid_thetas = theta[mask]
        valid_rhos = rho[mask]

        self.register_buffer('valid_thetas', valid_thetas)
        self.register_buffer('valid_rhos', valid_rhos)
        self.register_buffer('mask_flat', mask.view(-1))
        self.m = len(valid_thetas)

        # Precompute rotation grid for spatial alignment (optional)
        if return_aligned:
            self._init_rotation_grid(H, W)

    def _init_rotation_grid(self, H, W):
        # Create normalized coordinate grid [-1, 1]
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs)  # [H, W]
        self.register_buffer('base_grid', torch.stack([grid_x, grid_y], dim=-1))  # [H, W, 2]

    def _rotate_images(self, x, theta_e):
        """
        Rotate input x by -theta_e to align with horizontal direction.
        x: [B, 1, H, W]
        theta_e: [B]
        Returns: [B, 1, H, W]
        """
        B, _, H, W = x.shape

        # Construct rotation matrix (counter-clockwise positive; rotate clockwise by -theta_e)
        cos = torch.cos(-theta_e)  # [B]
        sin = torch.sin(-theta_e)  # [B]

        # Rotation matrix: [B, 2, 2]
        rot_mat = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=1)  # [B, 2, 2]

        # Expand base grid to batch dimension
        grid = self.base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        grid = grid.reshape(B, H * W, 2)  # [B, HW, 2]

        # Apply rotation: [B, HW, 2] = [B, HW, 2] @ [B, 2, 2]
        rotated_grid = torch.bmm(grid, rot_mat)  # [B, HW, 2]
        rotated_grid = rotated_grid.reshape(B, H, W, 2)  # [B, H, W, 2]

        # grid_sample expects grid ∈ [-1,1] (already normalized)
        x_rotated = F.grid_sample(
            x,
            rotated_grid,
            mode='bilinear',
            padding_mode='border',  # Alternatively 'zeros'
            align_corners=True
        )
        return x_rotated

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1 and H == 7 and W == 7, f"Expected [B,1,7,7], got {x.shape}"

        # 1. Compute rFFT
        x_rfft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')  # [B, 1, 7, 4]
        magnitude = x_rfft.abs() + 1e-8  # [B, 1, 7, 4]

        # 2. Extract valid frequency points
        mag_flat = magnitude.view(B, -1)  # [B, 28]
        mag_valid = mag_flat[:, self.mask_flat]  # [B, m]

        rho_flat = self.valid_rhos.to(x.device)  # [m]
        weighted_energy = mag_valid * rho_flat.unsqueeze(0)  # [B, m]

        # 3. Estimate principal orientation
        thetas = self.valid_thetas.to(x.device)  # [m]
        max_energy_idx = torch.argmax(weighted_energy, dim=1)  # [B]
        theta_e = thetas[max_energy_idx]  # [B]

        # 4. Angle normalization & sorting (for rot_inv_mag)
        theta_prime = (thetas.unsqueeze(0) - theta_e.unsqueeze(1) + math.pi) % math.pi  # [B, m]
        u = rho_flat.unsqueeze(0) * torch.cos(theta_prime)  # [B, m]
        v = rho_flat.unsqueeze(0) * torch.sin(theta_prime)  # [B, m]
        sort_key = u * 1e6 + v
        sorted_indices = torch.argsort(sort_key, dim=1)  # [B, m]
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
        rot_inv_mag = mag_valid[batch_indices, sorted_indices]  # [B, m]

        # 5. Generate aligned spatial image (optional)
        x_aligned = None
        if self.return_aligned:
            x_aligned = self._rotate_images(x, theta_e)

        if self.return_aligned:
            return x_aligned, theta_e
        else:
            return rot_inv_mag, theta_e

@MODELS.register_module()
class FAAHead(RotatedShared2FCBBoxHead):
    """Head supporting le90 angle encoding + Ere dimensionality reduction + angle alignment loss"""

    def __init__(self, *args,  **kwargs):
        super(FAAHead, self).__init__(*args, **kwargs)

        # faa module
        self.fam = FAA()
        self.m = self.fam.m  # Number of valid frequency points, e.g., 27

        self.gamma = nn.Parameter((1e-5) * torch.ones(1))

        # Modify input dimension: original spatial features 256*49 + FAM features 16 * self.m
        old_in_features = self.in_channels * self.roi_feat_area

        # Rebuild shared_fcs
        num_shared_fcs = len(self.shared_fcs)
        self.shared_fcs = nn.ModuleList()
        for i in range(num_shared_fcs):
            if i == 0:
                self.shared_fcs.append(nn.Linear(old_in_features, self.fc_out_channels + self.in_channels))
            else:
                self.shared_fcs.append(nn.Linear(self.fc_out_channels + self.in_channels, self.fc_out_channels))

        for m in self.shared_fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)

        N, C, H, W = x.shape

        # Apply FAM to each of the 256 channels independently
        fam_Ere_list = []
        fam_theta_e_list = []
        for i in range(256):
            channel_feat = x[:, i:i + 1, :, :]
            Ere, theta_e = self.fam(channel_feat)
            fam_Ere_list.append(Ere)
            fam_theta_e_list.append(theta_e)

        # Concatenate Ere and reduce dimensionality
        fam_Ere = torch.stack(fam_Ere_list, dim=1).view(N, -1)  # [N, 16*m*4]

        # Original spatial features
        spatial_feat = x.view(N, -1)  # [N, 256*49]

        # Feature fusion
        x = spatial_feat + self.gamma * fam_Ere

        # FC
        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred