import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import torch
import numpy as np
import open3d as o3d
from utils.helpers_3d import read_pcl, get_nndf_patches, build_patch_coords, get_3dtensors_patches
from utils.containers_3d import PclArray


def test_build_patch_coords():
    test_tensor = np.random.random((100, 100, 100))
    w = 3
    test_centers = np.random.randint(w // 2 + 1, 100, 10 * 3).reshape(10, 3)

    comparison = get_3dtensors_patches(
        torch.tensor(test_tensor)[None], w, np.hstack([np.zeros((10, 1), dtype=np.int64), test_centers]))

    coords = build_patch_coords(test_centers, w)

    assert np.allclose(
        test_tensor[coords[:, 0], coords[:, 1], coords[:, 2]].reshape(10, 3, 3, 3), comparison)


def test_get_nndf_patches():
    BUNNY_PATH = '/scratch/home/aelphy/projects/sandbox_data/bunny/reconstruction/bun_zipper.ply'
    pcl = read_pcl(BUNNY_PATH)
    res = 32
    dims = [res] * 3
    q_shape = [2] * (int(np.log(res) / np.log(2)) * 3)
    pcl_array = PclArray([pcl], res)
    w = 3
    p1 = np.concatenate([np.array([0]), np.random.randint(res, size=len(dims))])
    p2 = np.concatenate([np.array([0]), np.random.randint(res, size=len(dims))])
    p3 = np.concatenate([np.array([0]), np.random.randint(res, size=len(dims))])
    
    patches1 = get_nndf_patches(pcl_array[0], w, np.stack([p1, p2, p3]))
    for i in range(w):
        for j in range(w):
            for k in range(w):
                for l, p in enumerate([p1, p2, p3]):
                    pcd = o3d.geometry.PointCloud()
                    coords = pcl_array[0].mins + (p[1:] + np.array([i - (w // 2), j - (w // 2), k - (w // 2)]) + 0.5) * pcl_array[0].voxel_size
                    pcd.points = o3d.utility.Vector3dVector(
                        coords.reshape(1, -1))
                    
                    assert patches1[l, i, j, k] == torch.tensor(
                        np.exp(
                            -np.array(o3d.geometry.PointCloud.compute_point_cloud_distance(pcd, pcl))))
