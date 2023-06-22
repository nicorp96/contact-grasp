import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.point_net_helper import PointNetHelper
from src.utils.pcl_operations import (
    group_points,
    farthest_point_sample,
    query_ball_point,
    euclidean_distance,
)


class PointNetSA(nn.Module):
    def __init__(
        self, n_points, qb_radius, n_sample, n_input_ch, mlp, group_all
    ) -> None:
        super(PointNetSA, self).__init__()
        self._n_points = n_points
        self._qb_radius = qb_radius
        self._n_sample = n_sample
        self._mlp_convs_2d = nn.ModuleList()
        self._mlp_batch_norms_2d = nn.ModuleList()
        self._group_all = group_all
        b_ch = n_input_ch
        for n_output_ch in mlp:
            self._mlp_convs_2d.append(nn.Conv2d(b_ch, n_output_ch, 1))
            self._mlp_batch_norms_2d.append(nn.BatchNorm2d(n_output_ch))
            b_ch = n_output_ch

    def forward(self, x_y_z, points):
        """Compute PointNet++ Set Abstraction Layer

        Args:
            x_y_z (torch.Tensor): input points position data, [B, C, N]
            points (_type_): input points data, [B, D, N]

        Returns:
            new_x_y_z (torch.Tensor): sampled points position data, [B, C, N]
            new_points (torch.Tnesor): sample points feature data, [B, D', S]
        """
        x_y_z = x_y_z.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self._group_all:
            new_x_y_z, new_points = PointNetHelper.sample_and_group_all(
                x_y_z=x_y_z, points=points
            )
        else:
            new_x_y_z, new_points = PointNetHelper.sample_and_group(
                n_points=self._n_points,
                qb_radius=self._qb_radius,
                n_sample=self._n_sample,
                x_y_z=x_y_z,
                points=points,
            )

        new_points = new_points.permute(0, 3, 2, 1).float()
        for i in range(len(self._mlp_convs_2d)):
            conv = self._mlp_convs_2d[i]
            bn = self._mlp_batch_norms_2d[i]
            new_p_c = conv(new_points)
            new_p_b = bn(new_p_c)
            new_points = F.relu(new_p_b)

        new_points = torch.max(new_points, 2)[0]
        new_x_y_z = new_x_y_z.permute(0, 2, 1)

        return new_x_y_z, new_points


class PointNetSAMsg(nn.Module):
    def __init__(
        self, n_points, qb_radius_list, n_sample_list, n_input_ch, mlp_list
    ) -> None:
        super(PointNetSAMsg, self).__init__()
        self._n_points = n_points
        self._qb_radius_list = qb_radius_list
        self._n_sample_list = n_sample_list
        self._mlp_convs_2d = nn.ModuleList()
        self._mlp_batch_norms_2d = nn.ModuleList()
        for i in range(len(mlp_list)):
            conv_2d_list = nn.ModuleList()
            batch_norm_list = nn.ModuleList()
            b_ch = n_input_ch  # + 3
            for n_output_ch in mlp_list[i]:
                conv_2d_list.append(nn.Conv2d(b_ch, n_output_ch, 1))
                batch_norm_list.append(nn.BatchNorm2d(n_output_ch))
                b_ch = n_output_ch
            self._mlp_convs_2d.append(conv_2d_list)
            self._mlp_batch_norms_2d.append(batch_norm_list)

    def forward(self, x_y_z, points):
        x_y_z = x_y_z.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        batch = x_y_z.shape[0]
        ch = x_y_z.shape[2]

        new_x_y_z = group_points(x_y_z, farthest_point_sample(x_y_z, self._n_points))
        new_points_list = []
        for i in range(len(self._qb_radius_list)):
            group_idx = query_ball_point(
                qb_radius=self._qb_radius_list[i],
                n_sample=self._n_sample_list[i],
                x_y_z_1=x_y_z,
                x_y_z_2=new_x_y_z,
            )
            grouped_x_y_z = group_points(x_y_z, group_idx)
            grouped_x_y_z -= new_x_y_z.view(batch, self._n_points, 1, ch)
            if points is not None:
                grouped_points = group_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_x_y_z], dim=-1)
            else:
                grouped_points = grouped_x_y_z

            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.max(
                self.calculate_conv_batch_relu(
                    self._mlp_convs_2d[i],
                    self._mlp_batch_norms_2d[i],
                    grouped_points,
                ),
                2,
            )[0]
            new_points_list.append(new_points)

        new_x_y_z = new_x_y_z.permute(0, 2, 1)
        new_points_con = torch.cat(new_points_list, dim=1)
        return new_x_y_z, new_points_con

    @staticmethod
    def calculate_conv_batch_relu(conv_2d_list, bn_2d_list, grouped_points):
        grouped_points = grouped_points.to(dtype=torch.float32)
        for j in range(len(conv_2d_list)):
            grouped_points = F.relu(bn_2d_list[j](conv_2d_list[j](grouped_points)))
        return grouped_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x_y_z_1, x_y_z_2, points_1, points_2):
        """
        Input:
            x_y_z_1: input points position data, [B, C, N]
            x_y_z_2: sampled input points position data, [B, C, S]
            points_1: input points data, [B, D, N]
            points_2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        x_y_z_1 = x_y_z_1.permute(0, 2, 1)
        x_y_z_2 = x_y_z_2.permute(0, 2, 1)

        points_2 = points_2.permute(0, 2, 1)
        B, N, C = x_y_z_1.shape
        _, S, _ = x_y_z_2.shape

        if S == 1:
            interpolated_points = points_2.repeat(1, N, 1)
        else:
            dists = euclidean_distance(x_y_z_1, x_y_z_2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                group_points(points_2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points_1 is not None:
            points_1 = points_1.permute(0, 2, 1)
            new_points = torch.cat([points_1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = new_points.to(dtype=torch.float32)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
