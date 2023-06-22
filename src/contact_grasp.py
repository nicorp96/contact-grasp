import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
from IPython.core.debugger import set_trace
from torch.utils.data import DataLoader
from src.utils.point_net_layers import (
    PointNetFeaturePropagation,
    PointNetSA,
    PointNetSAMsg,
)
from src.contact_net_dataset import ContactNetPCDataset


class ContactGraspModel(nn.Module):
    def __init__(
        self,
        in_points=3000,
        in_ch=6,
        batch_size=10,
        pos_weight=3,
        pos_weight_2=6,
        top_k=512,
    ) -> None:
        super(ContactGraspModel, self).__init__()
        self.in_shape = (batch_size, in_ch, in_points)
        self._pos_weight = pos_weight
        self._pos_weight_2 = pos_weight_2
        self._top_k = top_k
        self._set_abstraction_1 = PointNetSAMsg(
            n_points=1000,
            qb_radius_list=[0.001, 0.002, 0.004],  # [0.01, 0.02, 0.04]
            n_sample_list=[32, 64, 128],
            mlp_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ],
            n_input_ch=3 + 3,
        )
        self._set_abstraction_2 = PointNetSAMsg(
            n_points=512,
            qb_radius_list=[0.004, 0.006, 0.008],  # [0.01, 0.02, 0.04]
            n_sample_list=[64, 64, 128],
            mlp_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ],
            n_input_ch=320 + 3,
        )
        self._set_abstraction_3 = PointNetSA(
            n_points=None,
            qb_radius=None,
            n_sample=None,
            n_input_ch=640 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self._feature_propagation_3 = PointNetFeaturePropagation(
            in_channel=1664, mlp=[256, 256]
        )
        self._feature_propagation_2 = PointNetFeaturePropagation(
            in_channel=576, mlp=[256, 128]
        )
        self._feature_propagation_1 = PointNetFeaturePropagation(
            in_channel=134, mlp=[128, 128]
        )
        self._conv1d_1 = nn.Conv1d(128, 128, 1)
        self._batch_norm_1 = nn.BatchNorm1d(128)
        self._dropout_1 = nn.Dropout(0.5)
        self._conv1d_2 = nn.Conv1d(128, 1, 1)

        self._conv1d_3 = nn.Conv1d(128, 128, 1)
        self._batch_norm_2 = nn.BatchNorm1d(128)
        self._dropout_2 = nn.Dropout(0.5)
        self._conv1d_4 = nn.Conv1d(128, 1, 1)

    def forward(self, X):
        x, y_true = X
        l0_points = x[:, 3:, :]
        l0_xyz = x[:, :3, :]
        l1_xyz, l1_points = self._set_abstraction_1(l0_xyz, l0_points)

        l2_xyz, l2_points = self._set_abstraction_2(l1_xyz, l1_points)

        l3_xyz, l3_points = self._set_abstraction_3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self._feature_propagation_3(l2_xyz, l3_xyz, l2_points, l3_points)

        l1_points = self._feature_propagation_2(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = self._feature_propagation_1(
            l0_xyz,
            l1_xyz,
            torch.cat([l0_xyz, x[:, 3:, :]], 1),
            l1_points,
        )

        # FC layers
        head_1 = torch.relu(self._batch_norm_1(self._conv1d_1(l0_points)))

        head_1 = self._dropout_1(head_1)
        head_1 = self._conv1d_2(head_1)
        head_1 = torch.sigmoid(head_1)
        head_2 = torch.relu(self._batch_norm_2(self._conv1d_3(l0_points)))
        head_2 = self._dropout_2(head_2)
        head_2 = self._conv1d_4(head_2)
        head_2 = torch.sigmoid(head_2)

        head_1 = head_1.permute(0, 2, 1)
        head_2 = head_2.permute(0, 2, 1)

        return head_1, head_2

    @staticmethod
    def dice_loss_fn(y_pred, y_true, eps=1):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        categ = torch.unique(y_true)
        numer = 0
        denom = 0
        for c in categ:
            idx = y_true == c
            y_tru = y_true[idx]
            y_hat = y_pred[idx]
            numer += torch.sum(y_hat == y_tru)
            denom += len(y_tru) + len(y_hat)
        return 1 - 2 * ((numer + eps) / (denom + eps))

    @staticmethod
    def focal_loss(y_pred, y_true, alpha, gamma):
        bce_loss = nnfunc.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    @staticmethod
    def top_k_loss(loss, top_k, dim=1):
        loss_k, _ = torch.topk(loss, top_k, dim=dim)
        loss_k_mean = torch.mean(loss_k)
        return loss_k_mean

    def loss_fn(self, y, y_pred, y_points, y_pred_points):
        weight_2 = torch.tensor(0.8, device=y.device)
        num_points = y.shape[1]
        weight_pos_1 = torch.ones([num_points, 1], device=y.device) * self._pos_weight
        weight_pos_2 = torch.ones([num_points, 1], device=y.device) * self._pos_weight_2
        loss_b_1 = nnfunc.binary_cross_entropy_with_logits(
            y_pred, y, weight=weight_2, pos_weight=weight_pos_1
        )

        loss_d_1 = self.dice_loss_fn(y_pred, y)
        loss_b_2 = nnfunc.binary_cross_entropy_with_logits(
            y_pred_points,
            y_points,
            weight=weight_2,
            pos_weight=weight_pos_2,
            # reduction="none", # for top_k calculation
        )
        loss_d_2 = self.dice_loss_fn(y_pred_points, y_points)

        # head_2_loss = self.top_k_loss(loss_b_2, self._top_k)
        # loss = loss_b_1 + loss_d_1 + head_2_loss
        # loss = loss_b_1 + loss_b_2

        loss = loss_b_1 + loss_d_1 + loss_b_2 + loss_d_2
        return loss


if __name__ == "__main__":
    n = 1
    point_cloud_data_set = ContactNetPCDataset("data_contact_points_4000")
    contact_net = ContactGraspModel()
    contact_net = contact_net.to(device="cuda:0")
    data_loader = DataLoader(point_cloud_data_set, batch_size=10, shuffle=True)
    for point, color, label, contact_points in data_loader:
        point = point.to(device="cuda:0")
        label = label.to(device="cuda:0")

        point = point.permute(0, 2, 1)
        # label = label.permute(0, 2, 1)
        # print(f"point = {point.shape}")
        # print(f"contact = {contact.shape}")
        X = point, color
        head_1, head_2 = contact_net(X)
        # loss = contact_net.loss_fn(label, y_pred, point)
        # print(loss)
        # visualize_point_cloud_from_numpy(point[0])
