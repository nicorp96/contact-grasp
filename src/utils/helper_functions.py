import open3d as o3d
import json
import numpy as np


def visualize_point_cloud_from_numpy(points, colors=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


def visualize_multiple_point_cloud_numpy(point, label):
    for i in range(point.shape[0]):
        xyz = point[i, :, :3]
        print(label.shape[1])
        for j in range(label.shape[1]):
            label_i = label[i, j, :]
            idx = label_i == 1
            point_g = xyz[idx]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyz)
            point_cloud.paint_uniform_color([0, 0, 0])
            pcl2 = o3d.geometry.PointCloud()
            pcl2.points = o3d.utility.Vector3dVector(point_g)
            pcl2.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([point_cloud, pcl2])


def open_json(config_path):
    with open(config_path) as config_file:
        data = json.loads(config_file.read())
    return data


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / m
    return points


def rotate_point_cloud_random(points):
    angle = np.random.uniform(-np.pi, np.pi)
    rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(points, rot_z)
