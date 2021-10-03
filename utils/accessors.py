import os
import sys
import time
import torch
import tntorch
import numpy as np
from typing import Iterable, Callable, Optional, Union
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.shape import z_coords2full, q_coords2full
from utils.helpers_3d import get_3dtensors_patches
from utils.helpers_2d import get_imgs_patches
from utils.misc import iterate_minibatches
from utils.tt_approx import idxmapper


class State:
    current: int
    start: int
    end: int
    total: int

    def __init__(
            self,
            current: Optional[int] = -1,
            start: Optional[int] = -1,
            end: Optional[int] = -1,
            total: Optional[int] = -1):

        self.current = current
        self.start = start
        self.end = end
        self.total = total


def build_no_enc_acc2d_(
        im: torch.Tensor,
        q_shape: Iterable,
        full_shape: Iterable,
        coords2full: Callable):

    def no_enc_acc(q_coords: torch.Tensor):
        full_coords = coords2full(
            q_coords.detach().cpu().long().numpy(),
            np.array(q_shape),
            np.array(full_shape))
        return im[full_coords[:, 0], full_coords[:, 1]]

    return no_enc_acc


def build_no_enc_acc2d_z(
        im: torch.Tensor, q_shape: Iterable, full_shape: Iterable):
    return build_no_enc_acc2d_(im, q_shape, full_shape, z_coords2full)


def build_no_enc_acc2d(
        im: torch.Tensor, q_shape: Iterable, full_shape: Iterable):
    return build_no_enc_acc2d_(im, q_shape, full_shape, q_coords2full)


def build_acc3d_multidim(
        im: Union[torch.Tensor],
        enc: torch.nn.Module,
        q_shape: Iterable,
        full_shape: Iterable,
        detach: bool,
        batch_size: int,
        patch_func: Callable,
        arg_patch_func: Callable,
        state: Optional[State] = None):

    def enc_acc(q_coords):
        np_q_coords = q_coords.detach().cpu().long().numpy()
        full_coords = np.vstack(
            q_coords2full(
                np_q_coords,
                np.array(q_shape),
                np.array(full_shape)))

        t0 = time.time()
        patches = patch_func(
            arg_patch_func(im),
            enc.rec_field,
            np.hstack(
                [np.zeros((len(q_coords), 1), dtype=np.int64),
                 full_coords[:, :-1]]))
        patches = patches.reshape(
            -1, enc.rec_field, enc.rec_field, enc.rec_field)

        res = []

        for idxs in iterate_minibatches(
                len(patches), batch_size, shuffle=False):
            res.append(
                enc(
                    patches[idxs].reshape(
                        len(idxs), 1,
                        enc.rec_field, enc.rec_field, enc.rec_field)
                ).cpu().reshape(len(idxs), -1)[
                    np.arange(len(idxs)), full_coords[idxs][:, -1]])

            if detach:
                res[-1] = res[-1].detach()
            else:
                if state is not None:
                    if state.current >= state.start and \
                            state.current < state.end:
                        state.current += len(idxs)
                    else:
                        res[-1] = res[-1].detach()

        return torch.cat(res)
    return enc_acc


def build_enc_acc2d_(
        im: torch.Tensor,
        enc: torch.nn.Module,
        q_shape: Iterable,
        full_shape:Iterable,
        detach: bool,
        batch_size: int,
        coords2full: Callable,
        state: Optional[State] = None):

    def enc_acc(q_coords):
        full_coords = np.vstack(
            coords2full(
                q_coords.detach().cpu().long().numpy(),
                np.array(q_shape),
                np.array(full_shape)))

        patches = get_imgs_patches(
            im[None],
            enc.rec_field,
            np.hstack([np.zeros((len(q_coords), 1), dtype=np.int64), full_coords]))
        
        res = []
        for idxs in iterate_minibatches(len(patches), batch_size, shuffle=False):
            res.append(
                    enc(
                        patches[idxs].reshape(len(idxs), 1, enc.rec_field, enc.rec_field).to(im.device)
                    ).cpu().reshape(-1))

            if detach:
                res[-1] = res[-1].detach()
            else:
                if state is not None:
                    if state.current >= state.start and state.current < state.end:
                        state.current += len(idxs)
                    else:
                        res[-1] = res[-1].detach()

        return torch.cat(res)

    return enc_acc


def build_enc_acc2d_multidim(
        im: torch.Tensor,
        enc: torch.nn.Module,
        q_shape: Iterable,
        full_shape:Iterable,
        detach: bool,
        batch_size: int,
        patch_func: Callable,
        arg_patch_func: Callable,
        state: Optional[State] = None):

    def enc_acc(q_coords):
        full_coords = np.vstack(
            q_coords2full(
                q_coords.detach().cpu().long().numpy(),
                np.array(q_shape),
                np.array(full_shape)))
        
        patches = patch_func(
            arg_patch_func(im),
            enc.rec_field,
            np.hstack(
                [np.zeros((len(q_coords), 1), dtype=np.int64),
                 full_coords[:, :-1]]))

        res = []
        
        for idxs in iterate_minibatches(len(patches), batch_size, shuffle=False):
            res.append(
                    enc(
                        patches[idxs].reshape(len(idxs), 1, enc.rec_field, enc.rec_field).to(im.device)
                    ).cpu().reshape(len(idxs),-1)[
                        np.arange(len(idxs)), full_coords[idxs][:, -1]])

            if detach:
                res[-1] = res[-1].detach()
            else:
                if state is not None:
                    if state.current >= state.start and state.current < state.end:
                        state.current += len(idxs)
                    else:
                        res[-1] = res[-1].detach()
        
        return torch.cat(res)
        
    return enc_acc


def build_enc_acc2d_z(im, enc, z_shape, full_shape, detach, batch_size, state=None):
    return build_enc_acc2d_(im, enc, z_shape, full_shape, detach, batch_size, z_coords2full, state)


def build_enc_acc2d(im, enc, q_shape, full_shape, detach, batch_size, state=None):
    return build_enc_acc2d_(im, enc, q_shape, full_shape, detach, batch_size, q_coords2full, state)


def build_no_enc_acc3d_(
        im: torch.Tensor,
        q_shape: Iterable,
        full_shape: Iterable,
        coords2full: Callable):

    def no_enc_acc(q_coords):
        full_coords = coords2full(
            q_coords.detach().cpu().long().numpy(),
            np.array(q_shape),
            np.array(full_shape))
        return im[full_coords[:, 0], full_coords[:, 1], full_coords[:, 2]]

    return no_enc_acc


def build_no_enc_acc3d(im: torch.Tensor, q_shape: Iterable, full_shape: Iterable):
    return build_no_enc_acc3d_(im, q_shape, full_shape, q_coords2full)


def build_acc3d_(
    im: torch.Tensor,
    enc: torch.nn.Module,
    q_shape: Iterable,
    full_shape: Iterable,
    detach: bool,
    batch_size: int,
    patch_func: Callable,
    arg_patch_func: Callable,
    state: Optional[State] = None):

    def enc_acc(q_coords):
        np_q_coords = q_coords.detach().cpu().long().numpy()
        full_coords = np.vstack(
            q_coords2full(
                np_q_coords,
                np.array(q_shape),
                np.array(full_shape)))

        t0 = time.time()
        patches = patch_func(
            arg_patch_func(im),
            enc.rec_field,
            np.hstack([np.zeros((len(q_coords), 1), dtype=np.int64), full_coords]))
        patches = patches.reshape(-1, enc.rec_field, enc.rec_field, enc.rec_field)

        res = []

        for idxs in iterate_minibatches(len(patches), batch_size, shuffle=False):
            res.append(
                enc(
                    patches[idxs].reshape(len(idxs), 1, enc.rec_field, enc.rec_field, enc.rec_field).cuda()
                ).cpu().reshape(-1)
            )

            if detach:
                res[-1] = res[-1].detach()
            else:
                if state is not None:
                    if state.current >= state.start and state.current < state.end:
                        state.current += len(idxs)
                    else:
                        res[-1] = res[-1].detach()

        return torch.cat(res)
    return enc_acc


def build_enc_acc3d(
    im: torch.Tensor,
    enc: torch.nn.Module,
    q_shape: Iterable,
    full_shape: Iterable,
    detach: bool,
    batch_size: int,
    state: Optional[State] = None):

    arg_patch_func = lambda x: x[None, ...].cpu()
    return build_acc3d_(im, enc, q_shape, full_shape, detach, batch_size, get_3dtensors_patches, arg_patch_func, state)


def evaluate(
    rec: tntorch.Tensor,
    decoder: torch.nn.Module,
    coords: torch.Tensor,
    dims: Iterable,
    w: int = 1):
    '''
    Extract patches from compressed qtt tensor and squeeze them through decoder.

    rec: qtt tensor
    decoder: torch model
    coords: centers of the patches
    w: patch size
    '''

    patches = torch.cat(
        [
            idxmapper(
                rec[batch_el],
                coords[coords[:, 0] == batch_el][:, 1:],
                dims,
                decoder.rec_field + w - 1
            )
            for batch_el in range(rec.shape[0])
        ]
    )

    dec = decoder(
        patches.reshape([-1, 1] + [decoder.rec_field + w - 1] * len(dims)).cuda()
    ).detach().cpu()

    return dec


def evaluate_no_dec(
        rec: tntorch.Tensor,
        coords: torch.Tensor,
        dims: Iterable,
        w: int = 1):
    patches = torch.cat(
        [
            idxmapper(
                rec[batch_el],
                coords[coords[:, 0] == batch_el][:, 1:],
                dims,
                w
            )
            for batch_el in range(rec.shape[0])
        ]
    )
    
    return patches
