""" Based on repo by Xu Yan
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

import torch
from src.utils.pcl_operations import (
    query_ball_point,
    farthest_point_sample,
    group_points,
)


class PointNetHelper:
    @staticmethod
    def sample_and_group(n_points, qb_radius, n_sample, x_y_z, points, use_x_y_z=True):
        """Sample and groups the points

        Args:
            n_points (_type_): number of points
            qb_radius (_type_): radius for query ball point calculation
            n_sample (_type_): number of sample
            x_y_z (_type_): input points positions
            points (_type_): input points data
            use_x_y_z (bool, optional): Use position xyz. Defaults to True.

        Returns:
            new_x_y_z, new_points: new points positions and points data
        """
        batch = x_y_z.shape[0]
        ch = x_y_z.shape[2]
        n_p = n_points
        fsp_idx = farthest_point_sample(x_y_z, n_points)  # [B, npoint, C]
        new_x_y_z = group_points(x_y_z, fsp_idx)  #  (batch_size, n_point, 3)
        idx = query_ball_point(qb_radius, n_sample, x_y_z, new_x_y_z)
        grouped_x_y_z = group_points(
            x_y_z, idx
        )  # (batch_size, n_point, n_sample, 3) / [batch, n_point, n_sample, ch]
        grouped_x_y_z_norm = grouped_x_y_z - new_x_y_z.view(batch, n_p, 1, ch)

        if points is not None:
            grouped_points = group_points(
                points, idx
            )  # (batch_size, n_point, n_sample, channel)
            if use_x_y_z:
                new_points = torch.cat(
                    [grouped_x_y_z_norm, grouped_points], dim=-1
                )  # (batch_size, n_point, n_sample, 3+channel) / [B, npoint, nsample, C+D]
            else:
                new_points = grouped_points
        else:
            new_points = grouped_x_y_z_norm

        return new_x_y_z, new_points

    @staticmethod
    def sample_and_group_all(x_y_z, points):
        """

        Args:
            x_y_z (_type_): input points positions
            points (_type_): input points data

        Returns:
            new_x_y_z, new_points: new points positions and points data
        """
        device = x_y_z.device
        batch = x_y_z.shape[0]
        n_ds = x_y_z.shape[1]
        ch = x_y_z.shape[2]

        new_x_y_z = torch.zeros(batch, 1, ch).to(device)  # (batch_size, 1, 3) /
        grouped_x_y_z = x_y_z.view(
            batch, 1, n_ds, ch
        )  # (batch_size, n_point=1, n_sample, 3)
        if points is not None:
            new_points = torch.cat(
                [grouped_x_y_z, points.view(batch, 1, n_ds, -1)], dim=-1
            )
        else:
            new_points = grouped_x_y_z
        return new_x_y_z, new_points
