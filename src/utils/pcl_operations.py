import torch


def euclidean_distance(source_p, target_p):
    batch = source_p.shape[0]
    n_ds = source_p.shape[1]
    n_p = target_p.shape[1]
    distance = -2 + torch.matmul(source_p, target_p.permute(0, 2, 1))
    distance += torch.sum(source_p**2, -1).view(batch, n_ds, 1)
    distance += torch.sum(target_p**2, -1).view(batch, 1, n_p)
    return distance


def query_ball_point(qb_radius, n_sample, x_y_z_1, x_y_z_2):
    device = x_y_z_1.device
    batch = x_y_z_1.shape[0]
    n_ds = x_y_z_1.shape[1]
    n_p = x_y_z_2.shape[1]
    idx = (
        torch.arange(n_ds, dtype=torch.long)
        .to(device)
        .view(1, 1, n_ds)
        .repeat([batch, n_p, 1])
    )
    distance = euclidean_distance(x_y_z_2, x_y_z_1)
    idx[distance > qb_radius**2] = n_ds
    idx = idx.sort(dim=-1)[0][:, :, :n_sample]
    idx_1 = idx[:, :, 0].view(batch, n_p, 1).repeat([1, 1, n_sample])
    mask = idx == n_ds
    idx[mask] = idx_1[mask]
    return idx


def group_points(points, idx):
    device = points.device
    batch = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = (
        torch.arange(batch, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_idx, idx, :]
    return new_points


def farthest_point_sample(x_y_z, n_points):
    device = x_y_z.device
    batch = x_y_z.shape[0]
    n_ds = x_y_z.shape[1]
    c = x_y_z.shape[2]
    centroids = torch.zeros(batch, n_points, dtype=torch.long).to(device)
    distance = torch.ones(batch, n_ds, dtype=torch.float64).to(device) * 1e10
    farthest = torch.randint(0, n_ds, (batch,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch, dtype=torch.long).to(device)
    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = x_y_z[batch_indices, farthest, :].view(batch, 1, 3)
        dist = torch.sum((x_y_z - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
