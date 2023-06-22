import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from IPython.core.debugger import set_trace
from src.utils.helper_functions import (
    visualize_point_cloud_from_numpy,
    normalize_point_cloud,
    rotate_point_cloud_random,
)


class ContactNetPCDataset(Dataset):
    POINT_STRING = "."
    UNDERSCORE_STRING = "_"

    def __init__(
        self,
        data_dir,
        format_data="npz",
        mode="handoff",
        rotation=False,
        scaling=False,
        noise=False,
    ):
        super(ContactNetPCDataset, self).__init__()
        self._rotation = rotation
        self._scaling = scaling
        self._noise = noise
        data_dir = os.path.expanduser(os.path.join(data_dir, "train"))
        self._file_names = OrderedDict()
        for file_name in next(os.walk(data_dir))[-1]:
            object_name = file_name
            file_name = os.path.join(data_dir, file_name)
            self._file_names[object_name] = [file_name]

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        object_name = list(self._file_names.keys())[index]
        npz_file = np.load(self._file_names[object_name][0])
        points = npz_file["points"]
        contact_grasp_points = npz_file["contact_grasp_points"]
        label = npz_file["label"]
        colors = npz_file["colors"]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        point_cloud.normals = o3d.utility.Vector3dVector(points[:, 3:])
        point_cloud.paint_uniform_color([0, 0, 0])
        if self._rotation:
            # o3d.visualization.draw_geometries([point_cloud])
            # points = np.append(
            #     np.asarray(point_cloud.points), np.asarray(point_cloud.normals), 1
            # )
            if np.random.uniform(0, 1) > 1 - 0.25:
                points[:, :3] = rotate_point_cloud_random(points[:, :3])

        if self._scaling:
            scale_random = np.random.uniform(0.4, 1.0)
            point_cloud.scale(scale=scale_random, center=point_cloud.get_center())
            points = np.append(
                np.asarray(point_cloud.points), np.asarray(point_cloud.normals), 1
            )
        random_choice = np.random.choice(10, 1, replace=False)
        if self._noise and (random_choice % 2) == 1:
            points += np.random.normal(0.0, 0.01, points.shape)
            # # random augmentation colors
            # alpha = 1.0 + np.random.uniform(-brightness, brightness)
            # image *= alpha

        points[:, :3] = normalize_point_cloud(points[:, :3])

        return (
            points.astype(np.float64),
            colors.astype(np.float64),
            label.astype(np.float32),
            contact_grasp_points.astype(np.float32),
        )


class ContactNetPCDatasetTest(Dataset):
    POINT_STRING = "."
    UNDERSCORE_STRING = "_"

    def __init__(
        self,
        data_dir,
        format_data="npz",
        mode="handoff",
    ):
        super(ContactNetPCDatasetTest, self).__init__()
        data_dir = os.path.expanduser(os.path.join(data_dir, "test"))
        self._file_names = OrderedDict()
        for file_name in next(os.walk(data_dir))[-1]:
            session_name = file_name.split(self.UNDERSCORE_STRING)[0]
            sub_1 = session_name + self.UNDERSCORE_STRING + mode
            sub_2 = self.POINT_STRING + format_data
            index_1 = file_name.index(sub_1)
            index_2 = file_name.index(sub_2)
            obj_name = file_name[index_1 + len(sub_1) + 1 : index_2]
            file_path = os.path.join(data_dir, file_name)
            if "-" in obj_name:
                index_3 = obj_name.index("-")
                obj_name = obj_name[:index_3]
            if obj_name not in self._file_names:
                self._file_names[obj_name] = [file_path]
            else:
                self._file_names[obj_name].append(file_path)

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        object_name = list(self._file_names.keys())[index]
        points = []
        colors = []
        label = []
        contact_grasp_points = []

        for file_path in self._file_names[object_name]:
            npz_file = np.load(file_path)
            points.append(npz_file["points"].astype(np.float64))
            color_n = np.zeros(npz_file["colors"].shape, dtype=np.float64)  # * 0.1
            colors.append(color_n)
            label.append(npz_file["label"].astype(np.float32))
            contact_grasp_points.append(
                npz_file["contact_grasp_points"].astype(np.float32)
            )

        return (
            points,
            colors,
            label,
            contact_grasp_points,
        )


if __name__ == "__main__":
    point_cloud_data_set_test = ContactNetPCDatasetTest("data_contact_points_7000")
    point_cloud_data_set = ContactNetPCDataset(
        "data_contact_points_7000", rotation=False, noise=False
    )
    data_loader_test = DataLoader(point_cloud_data_set_test, batch_size=1, shuffle=True)
    data_loader = DataLoader(point_cloud_data_set, batch_size=1, shuffle=True)
    print(point_cloud_data_set_test._file_names.keys())

    for point, color, label, contact_points in data_loader:
        print(f"point = {point.shape}")
        print(f"colors shape = {color.shape}")
        print(f"label = {label.shape}")
        print(f"contact_points = {contact_points.shape}")
        visualize_point_cloud_from_numpy(point[0, :, :3], colors=color[0, :, :] * 0.0)

    for point, contact, label, contact_points in data_loader_test:
        print(f"point = {len(point)}")
        print(f"contact shape = {len(contact)}")
        print(f"label = {len(label)}")
    # visualize_point_cloud_from_numpy(point[0, :, :3]
