import os
import open3d as o3d
import torch
import numpy as np

from torchmetrics.classification import BinaryConfusionMatrix
from src.contact_net_dataset import (
    ContactNetPCDataset,
    ContactNetPCDatasetTest,
)


def visualize_test_from_label(
    x, y_pred, y_true, color_pred=[0, 1, 0], color_true=[1, 0, 0], debug=True
):
    x = x.cpu().numpy().squeeze()
    y_pred = y_pred.detach().cpu().numpy().squeeze()
    y_true = y_true.cpu().numpy().squeeze()
    th = (y_pred.max() + y_pred.min()) / 2
    # print(th)
    th = 0.7
    idx_pred = y_pred >= th
    point_cloud_pred = x[:, :3][idx_pred]

    idx_true = y_true == 1.0
    point_cloud = x[:, :3][idx_true]

    pc_pred = o3d.geometry.PointCloud()
    pc_pred.points = o3d.utility.Vector3dVector(point_cloud_pred)
    pc_pred.paint_uniform_color(color_pred)
    pc_true = o3d.geometry.PointCloud()
    pc_true.points = o3d.utility.Vector3dVector(point_cloud)
    pc_true.paint_uniform_color(color_true)

    pc_x = o3d.geometry.PointCloud()
    pc_x.points = o3d.utility.Vector3dVector(x[:, :3])
    pc_x.paint_uniform_color([0, 0, 0])

    if debug:
        print(y_pred.shape)
        print(y_true.shape)
        print(x.shape)
        print(y_pred.max())
        print(y_pred.min())
        print(y_true.max())
        print(y_true.min())
        o3d.visualization.draw_geometries([pc_true])
        o3d.visualization.draw_geometries([pc_pred])
        o3d.visualization.draw_geometries([pc_true, pc_pred])
        o3d.visualization.draw_geometries([pc_x, pc_pred])
        o3d.visualization.draw_geometries([pc_x, pc_true])
        o3d.visualization.draw_geometries([pc_x, pc_pred, pc_true])


def dice_coeff(y_pred, y_true, eps=1.0):
    y_true = y_true[0, :, 0]
    y_pred = y_pred[0, :, 0]

    categ = torch.unique(y_true)
    numer = 0
    denom = 0
    for c in categ:
        idx = y_true == c
        y_tru = y_true[idx]
        y_hat = y_pred[idx]
        numer += torch.sum(y_hat == y_tru)
        denom += len(y_tru) + len(y_hat)
    # idx = y_true == 1.0
    # y_tru = y_true[idx]
    # y_pre = y_pred[idx]
    # numer = torch.sum(y_pre == y_tru) + eps
    # denom = len(y_tru) + len(y_pre) + eps

    # return 2 * ((numer) / (denom))
    return 2 * ((numer + eps) / (denom + eps))


def compute_iou(y_pred, y_true):
    y_true = y_true[0, :, 0]
    y_pred = y_pred[0, :, 0]

    intersection = torch.sum(torch.abs(y_pred * y_true))  # true positives
    union = len(y_pred) + len(y_true) - intersection
    mean_iou = torch.mean((intersection + 1) / (union + 1))
    return mean_iou


def compute_error(y_pred, y_true):
    device = y_pred.device
    y_true = y_true[0, :, 0]
    y_pred = y_pred[0, :, 0]

    total_points = y_pred.shape[0]

    idx = y_true == 0.0
    y_true_neg = y_true[idx]
    y_pred_idx_neg = y_pred[idx]

    idx = y_true == 1.0

    y_true_pos = y_true[idx]
    y_pred_idx_pos = y_pred[idx]

    total_true = y_true_pos.shape[0]

    y_tn = torch.sum((y_pred_idx_neg < 0.7) == (y_true_neg == 0.0))
    y_tp = torch.sum((y_pred_idx_pos > 0.7) == (y_true_pos == 1.0))

    # print(y_tn)
    # print(y_tp)
    # print((total_points - y_tn - y_tp) / total_points)
    y_error = (total_points - y_tn - y_tp) / total_points
    # y_error = (1000 - y_tp) / 1000
    bcm = BinaryConfusionMatrix(threshold=0.7).to(device=device)
    conf_mat = bcm(y_pred, y_true)
    f_n = conf_mat[0, 1]
    f_p = conf_mat[1, 0]
    # print(y_error)
    y_error = (total_true - y_tp) / (total_true)
    error_per = y_error  # * 100
    return error_per


class ValidationContactGrasp:
    def __init__(
        self,
        model,
        path_weights,
        object_train_pred_show,
        path_data_set="data_contact_points",
        device="cuda:0",
        debug=False,
    ) -> None:
        self._path_weights = path_weights
        self._contact_grasp = model
        self._device = device
        self._contact_ds_train = ContactNetPCDataset(
            os.path.join(os.getcwd(), path_data_set),
        )
        self._contact_ds_val = ContactNetPCDatasetTest(
            os.path.join(os.getcwd(), path_data_set)
        )
        # self._data_loader_train_ds = torch.utils.data.DataLoader(
        #     self._contact_ds_train, batch_size=1
        # )
        # self._data_loader_val_ds = torch.utils.data.DataLoader(
        #     self._contact_ds_val, batch_size=1
        # )
        self._object_train_pred_show = object_train_pred_show
        self._contact_grasp_eval = self._load_weights()
        self._debug = debug

    def _load_weights(self):
        self._contact_grasp = self._contact_grasp.to(device=self._device)
        self._contact_grasp.load_state_dict(
            torch.load(self._path_weights)  # ["model_state_dict"]
        )
        contact_grasp_eval = self._contact_grasp.eval()
        return contact_grasp_eval

    @staticmethod
    def tensor_from_numpy(x, colors, label, contact_points, device):
        x = np.expand_dims(x, axis=0)
        colors = np.expand_dims(colors, axis=0)
        label = np.expand_dims(label, axis=0)
        contact_points = np.expand_dims(contact_points, axis=0)

        x = torch.from_numpy(x)
        colors = torch.from_numpy(colors)
        label = torch.from_numpy(label)
        contact_points = torch.from_numpy(contact_points)
        x = x.to(device=device, non_blocking=True)
        x = x.permute(0, 2, 1)
        label = label.to(device=device, non_blocking=True)
        label = label.permute(0, 2, 1)
        colors = colors.to(device=device, non_blocking=True)
        colors = colors.permute(0, 2, 1)
        contact_points = contact_points.to(device=device, non_blocking=True)
        contact_points = contact_points.permute(0, 2, 1)
        return x, colors, label, contact_points

    @staticmethod
    def visualize_predictions(
        x, y_true, contact_points, y_pred_1, y_pred_2, debug=False
    ):
        y_pred_1 = y_pred_1.permute(0, 2, 1)
        y_pred_2 = y_pred_2.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        # Visualize contact grasp prediction -> True:red, Pred: Green
        visualize_test_from_label(
            x,
            y_pred_1[0, :],
            y_true[0, :],
            color_pred=[0, 1, 0],
            color_true=[1, 0, 0],
            debug=debug,
        )

        # Visualize contact points prediction -> True:Blue, Pred: Green
        visualize_test_from_label(
            x,
            y_pred_2[0, :],
            contact_points[0, :],
            color_pred=[1, 0, 1],
            color_true=[0, 0, 1],
            debug=debug,
        )

    def evaluation_train_data(self):
        for index, (x, colors, label, contact_points) in enumerate(
            self._contact_ds_train
        ):
            object_name = list(self._contact_ds_train._file_names.keys())[index]
            if object_name in self._object_train_pred_show:
                x, colors, y_true, contact_points = self.tensor_from_numpy(
                    x, colors, label, contact_points, self._device
                )

                X = x, colors
                y_pred_1, y_pred_2 = self._contact_grasp_eval.forward(X)
                loss = self._contact_grasp_eval.loss_fn(
                    y_true, y_pred_1, contact_points, y_pred_2
                )
                self.visualize_predictions(
                    x, y_true, contact_points, y_pred_1, y_pred_2, debug=self._debug
                )
                print(f"loss = {loss.data:6.8f}")
                # score_head_1 = dice(y_pred_1 > 0.7, y_true == 1.0)
                # score_head_2 = dice(y_pred_2 > 0.7, contact_points == 1.0)
                score_head_1 = dice_coeff(y_pred_1, y_true)
                score_head_2 = dice_coeff(y_pred_2, contact_points)
                error_1 = compute_error(y_pred_1, y_true)
                error_2 = compute_error(y_pred_2, contact_points)
                print(f"dice score head 1 = {error_1:6.8f}")
                print(f"dice score head 2 = {error_2:6.8f}")

    def evaluation_test_data(self):
        for index, (x_list, colors_list, label_list, contact_points_list) in enumerate(
            self._contact_ds_val
        ):
            object_name = list(self._contact_ds_val._file_names.keys())[index]
            print(object_name)
            score_head_1 = torch.tensor(0, device=self._device, dtype=torch.float)
            score_head_2 = torch.tensor(0, device=self._device, dtype=torch.float)
            # if "cylinder_small" not in object_name:
            #     continue
            for i in range(len(x_list)):
                # if i > 3:
                #     continue
                x, colors, y_true, contact_points = self.tensor_from_numpy(
                    x_list[i],
                    colors_list[i],
                    label_list[i],
                    contact_points_list[i],
                    self._device,
                )

                X = x, colors
                y_pred_1, y_pred_2 = self._contact_grasp_eval.forward(X)
                loss = self._contact_grasp_eval.loss_fn(
                    y_true, y_pred_1, contact_points, y_pred_2
                )
                self.visualize_predictions(
                    x, y_true, contact_points, y_pred_1, y_pred_2, self._debug
                )
                # print(f"loss = {loss.data:6.8f}")
                error_1 = compute_error(y_pred_1, y_true)
                error_2 = compute_error(y_pred_2, contact_points)
                # score_head_1 += dice(y_pred_1 > 0.7, y_true == 1.0)
                # score_head_2 += dice(y_pred_2 > 0.7, contact_points == 1.0)
                score_head_1 += error_1
                score_head_2 += error_2
                # print(f"dice score head 1 = {error_1:6.8f}")
                # print(f"dice score head 2 = {error_2:6.8f}")
            print(len(x_list))
            avg_1 = score_head_1 / len(x_list)  # * 100
            avg_2 = score_head_2 / len(x_list)  # * 100
            print(f"dice score avg head 1 = {avg_1:6.8f}")
            print(f"dice score avg head 2 = {avg_2:6.8f}")
