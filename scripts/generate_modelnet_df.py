import torch
import click
import trimesh
import numpy as np
import kaolin as kl
import os.path as osp
import glob
import tqdm
from joblib import Parallel, delayed


def process_one_obj_mesh(path: str, resolution: int):
    assert path[-4:] == '.obj'

    basedir = osp.dirname(path)
    basename = osp.basename(path)
    new_path = osp.join(basedir, '{}_{}.df'.format(basename[:-4], resolution))

    if osp.exists(new_path):
        return

    mesh = trimesh.load(path)
    grid = np.arange(resolution)
    x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')
    x_n, y_n, z_n = x.ravel(), y.ravel(), z.ravel()
    coords = np.vstack([x_n.ravel(), y_n.ravel(), z_n.ravel()]).T

    xyzmin = mesh.vertices.min(0)
    xyzmax = mesh.vertices.max(0)
    max_size = np.max([np.max(xyzmax - xyzmin), 0])

    real_coords = xyzmin + (coords + 0.5) * max_size / resolution
    df = torch.exp(
        -kl.metrics.trianglemesh.point_to_mesh_distance(
            torch.tensor(real_coords)[None].float().cuda(),
            torch.tensor(mesh.vertices)[None].float().cuda(),
            torch.tensor(mesh.faces).cuda())[0]).detach().cpu()

    np.savetxt(
        new_path,
        df.numpy().ravel())


@click.command()
@click.option('-r', '--resolution', type=int, help='Resolution')
def main(resolution: int):
    files = glob.glob('../data/ModelNet10/raw/**/train/*.obj')
    Parallel(n_jobs=8)(
        delayed(process_one_obj_mesh)(f, resolution) for f in tqdm.tqdm(files))


if __name__ == '__main__':
    main()
