import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import configparser
import open3d as o3d
from torchmetrics.classification import BinaryAccuracy, BinaryMatthewsCorrCoef
from src.utils.trainer_contact import TrainerWithTest
from src.contact_grasp_fusion import ContactGraspModelFusion
from src.contact_net_dataset import ContactNetPCDataset, ContactNetPCDatasetTest
from src.utils.helper_functions import open_json


class MyTrainer(TrainerWithTest):
    def metrics(self, L, X, Y_pred, Y_true):
        y_pred_contact_pts, y_pred_cont = Y_pred
        label, contact_points = Y_true
        device = label.device

        label = label.permute(0, 2, 1)
        contact_points = contact_points.permute(0, 2, 1)

        label_pred = y_pred_cont.to(dtype=torch.float)
        contact_pred = y_pred_contact_pts.to(dtype=torch.float)
        label = label.to(dtype=torch.float)
        contact_points = contact_points.to(dtype=torch.float)

        acc_metric = BinaryAccuracy(threshold=0.5).to(device=device)
        mcc_metric = BinaryMatthewsCorrCoef().to(device=device)
        # Evaluation Head 1
        acc_1 = acc_metric(label_pred, label)
        mcc_1 = mcc_metric(label_pred, label)
        # Evaluation Head 2
        acc_2 = acc_metric(contact_pred, contact_points)
        mcc_2 = mcc_metric(contact_pred, contact_points)

        m_iou_1 = self.avg_class_Iou(label_pred, label)
        m_iou_2 = self.avg_class_Iou(contact_pred, contact_points)
        acc = (acc_1 + acc_2) / 2
        return {
            "loss": L,
            "acc": acc,
            "acc_1": acc_1,
            "m_iou_1": m_iou_1,
            "mcc_1": mcc_1,
            "acc_2": acc_2,
            "m_iou_2": m_iou_2,
            "mcc_2": mcc_2,
        }

    @staticmethod
    def avg_class_Iou(y_pred, y_true, threshold=0.5):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        correct_class_lbl = np.sum((y_pred > threshold) & (y_true == 1))
        iou_class = np.sum(((y_pred > threshold) | (y_true == 1)))

        m_iou = np.mean(
            np.array(correct_class_lbl) / (np.array(iou_class, dtype=np.float64) + 1e-6)
        )
        return m_iou

    def train_report(self, epoch, step):
        # avg_acc = (self.avg_acc_1 + self.avg_acc_2) / 2
        print(
            f"    {epoch:03d}/{step:05d}:  loss: {self.loss:6.8f} , acc: {self.avg_acc:6.4f}"  # , miou: {self.m_iou:6.4f}"
        )

    def test_report(self, epoch, step):
        print("--" * 50)
        print(
            f"> {epoch:03d}/{step+1:05d}:  loss {self.test_loss:6.8f}, acc {self.test_acc:6.4f}"
        )
        print("-" * 50)


class MainTrainingContactGraspFusion:
    NUM_SHOW_DS = 5
    FOLDER_VALIDATION = "evaluation_data"

    def __init__(self, config_path) -> None:
        self._config = configparser.ConfigParser()
        self._config.read_dict(open_json(config_path))
        self._device = "cuda:0"
        self._contact_net_pcl_ds_test = ContactNetPCDatasetTest(
            self._config["data_set"]["data_dir"]
        )
        self._contact_net_pcl_ds_train = ContactNetPCDataset(
            os.path.join(os.getcwd(), self._config["data_set"]["data_dir"]),
            rotation=True,
            scaling=False,
            noise=True,
        )
        self._data_loader_train = torch.utils.data.DataLoader(
            self._contact_net_pcl_ds_train,
            batch_size=self._config["data_set"].getint("batch_size"),
            shuffle=True,
            pin_memory=True,
            num_workers=self._config["data_set"].getint("num_workers"),
        )
        self._data_loader_test = torch.utils.data.DataLoader(
            self._contact_net_pcl_ds_test,
            batch_size=2,
        )
        self._contact_grasp = ContactGraspModelFusion(
            in_points=self._config["model"].getint("in_points"),
            in_ch=self._config["model"].getint("in_ch"),
            batch_size=self._config["data_set"].getint("batch_size"),
            pos_weight=self._config["model"].getint("pos_weight"),
            pos_weight_2=self._config["model"].getint("pos_weight_2"),
        )

    def visualize_random_ds(self, show=True):
        if show:
            for index in np.random.choice(
                len(self._contact_net_pcl_ds_train), self.NUM_SHOW_DS
            ):
                points, colors, label, contact_points = self._contact_net_pcl_ds_train[
                    index
                ]
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([point_cloud])

    def run_training(self):
        opt = torch.optim.Adam(
            self._contact_grasp.parameters(),
            lr=self._config["training"].getfloat("base_lr"),
            betas=(
                self._config["training"].getfloat("beta_1"),
                self._config["training"].getfloat("beta_2"),
            ),
        )
        cfg = {
            "n_epochs": self._config["training"].getint("max_epochs"),
            "batch_size": self._config["data_set"].getint("batch_size"),
            "opt": opt,
            "train_dl": self._data_loader_train,
            "test_dl": None,  # self._data_loader_test,
            "report_period": self._config["training"].getint("report_period"),
            "test_period": self._config["training"].getint("val_interval"),
            "n_save_weights": self._config["training"].getint("n_save_weights"),
            "dir_save_weights": os.path.join(
                os.getcwd(),
                self.FOLDER_VALIDATION,
                self._config["training"]["save_folder_name"],
            ),
            "device": self._device,
        }

        tr = MyTrainer(self._contact_grasp, cfg)
        tr.summary()
        hist = tr.train()
        torch.save(tr.mdl.state_dict(), "contact_grasp_fusion_weights.pth")

        tr.save_val("loss")
        tr.save_val("acc_1")
        tr.save_val("m_iou_1")
        tr.save_val("mcc_1")

        tr.save_val("acc_2")
        tr.save_val("m_iou_2")
        tr.save_val("mcc_2")

        plt.subplot(4, 1, 1)
        tr.plot_val("loss")
        plt.subplot(4, 1, 2)
        tr.plot_val("acc_1")
        plt.subplot(4, 1, 3)
        tr.plot_val("m_iou_1")
        plt.subplot(4, 1, 4)
        tr.plot_val("mcc_1")
        plt.show()
