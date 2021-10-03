import os
import sys

from numpy.lib.function_base import quantile
import torch
import pytorch_lightning as pl
import tntorch as tn
import numpy as np
from functools import partial
from typing import Iterable, Callable, Optional

sys.path.append('../')
sys.path.append('/scratch/lib')

from utils.radam import RAdam
from utils.svd import truncate_ca_qtt
from utils.tt_approx import get_train_features, get_test_features
from utils.tt_approx import merge_cores_to_batch, idxmapper
from utils.train import compute_CE, compute_MSE
from utils.shape import q_coords2full
from models.dense import Classifier, TwoHeadsRegressor, \
                         TwoHeadsRegressorNoBN, Regressor, \
                         Encoder, Decoder
from models.cnn import EncoderCA_3D
from models.resnet3d import generate_model


def do_nothing(x: torch.Tensor):
    return x


def add_dim(x: torch.Tensor):
    return x[None]


def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


def metric(preds, targets):
    sigma = preds[:, 2] - preds[:, 0]
    sigma[sigma < 70] = 70
    delta = (preds[:, 1] - targets).abs()
    delta[delta > 1000] = 1000
    return -np.sqrt(2) * delta / sigma - torch.log(np.sqrt(2) * sigma)


def accuracy(preds, targets):
    return (torch.argmax(preds, -1) == targets).sum() / float(len(targets))


class TTClassifier(pl.LightningModule):
    def __init__(
            self,
            resolution: Iterable[int],
            build_encoder_function: Callable,
            path_function: Callable,
            q_shape: Iterable[int],
            ndims: int,
            rank: int,
            num_classes: int,
            full_shape: Iterable[int],
            arg_patch_function: Optional[Callable] = do_nothing):
        super().__init__()
        build_enc_f = partial(
            build_encoder_function,
            arg_patch_func=arg_patch_function,
            patch_func=path_function)
        self.qtts = []
        self.rank = rank
        self.q_shape = q_shape
        self.resolution = resolution
        self.enc = EncoderCA_3D(num_out_channels=ndims)
        self.automatic_optimization = False

        self.encoder = partial(
            build_enc_f,
            enc=self.enc,
            q_shape=q_shape,
            full_shape=full_shape,
            detach=False,
            batch_size=10000)
        self.classifier = Classifier(rank, num_classes)

    def forward(self, x):
        if len(self.qtts) == 0:
            return

        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)

        rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        _, stack = get_train_features(
            merge_cores_to_batch(self.qtts),
            self.rank)
        features_val = get_test_features(rec, stack)

        return self.classifier(features_val)

    def training_step(self, batch, batch_idx):
        x, y = batch
        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)

        rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        detached_rec = tn.Tensor(
            [core.detach() for core in rec.cores], batch=True)

        try:
            _, stack = get_train_features(
                merge_cores_to_batch(self.qtts + [rec]), self.rank)
        except:
            print('failed stack')
            return

        features_train = get_test_features(rec, stack)
        preds = self.classifier(features_train.to(y.device))

        self.log('train_acc', accuracy(preds, y))

        loss = compute_CE(y, preds)
        self.log('train_loss', loss)
        self.trainer.train_loop.running_loss.append(loss)

        opt = self.optimizers()

        old_weights = [self.enc.state_dict(), self.classifier.state_dict()]
        try:
            self.manual_backward(loss, opt)
            opt.step()
            opt.zero_grad()
            self.qtts.append(detached_rec)
        except:
            import traceback
            traceback.print_exc()
            self.enc.load_state_dict(old_weights[0])
            self.classifier.load_state_dict(old_weights[1])
            self.log('failed back prop', 1)
            return {
                'preds': preds,
                'labels': y,
                'loss': loss}

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def on_train_epoch_start(self):
        print('Starting new epoch, num qtts: {}'.format(len(self.qtts)))
        self.qtts = []

    def validation_step(self, val_batch, batch_idx):
        if len(self.qtts) == 0:
            return

        x, y = val_batch
        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)

        rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        _, stack = get_train_features(
            merge_cores_to_batch(self.qtts), self.rank)

        features_val = get_test_features(rec, stack)
        preds = self.classifier(features_val.to(y.device))

        self.log('val_acc', accuracy(preds, y))

        loss = compute_CE(y, preds)
        self.log('val_loss', loss)

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=1e-3)


class TTQuantileRegressor(pl.LightningModule):
    def __init__(
            self,
            resolution: Iterable[int],
            build_encoder_function: Callable,
            use_encoder: bool,
            path_function: Callable,
            q_shape: Iterable[int],
            ndims: int,
            rank: int,
            full_shape: Iterable[int],
            use_extra_features: bool,
            quantiles: Iterable[float]):
        super().__init__()
        if use_encoder:
            build_enc_f = partial(
                build_encoder_function,
                arg_patch_func=add_dim,
                patch_func=path_function)
        else:
            build_enc_f=build_encoder_function

        self.qtts = []
        self.rank = rank
        self.q_shape = q_shape
        self.resolution = resolution
        self.quantiles = quantiles
        self.automatic_optimization = False
        self.use_encoder = use_encoder
        if use_encoder:
            self.enc = EncoderCA_3D(num_out_channels=ndims)
            self.encoder = partial(
                build_enc_f,
                enc=self.enc,
                q_shape=q_shape,
                full_shape=full_shape,
                detach=False,
                batch_size=10000)
        else:
            self.encoder = partial(
                build_enc_f,
                q_shape=q_shape,
                full_shape=full_shape)
        if use_extra_features:
            self.regressor = TwoHeadsRegressorNoBN(8, rank + 100, 3).cuda()
        else:
            self.regressor = TwoHeadsRegressorNoBN(1, rank + 100, 3).cuda()

    def forward(self, x1, x2):
        if len(self.qtts) == 0:
            print('Empty stack')
            return

        accs = [None] * len(x1)
        for i, im in enumerate(x1):
            accs[i] = self.encoder(im=im)

        rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        _, stack = get_train_features(
            merge_cores_to_batch(self.qtts), self.rank)
        features_val = get_test_features(rec, stack)

        return self.regressor(features_val, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y, init_values = batch
        accs = [None] * len(x1)
        for i, im in enumerate(x1):
            accs[i] = self.encoder(im=im)

        try:
            rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
            detached_rec = tn.Tensor(
                [core.detach() for core in rec.cores], batch=True)
        except:
            print('failed qtt')
            return

        try:
            _, stack = get_train_features(
                merge_cores_to_batch(self.qtts + [detached_rec]), self.rank)
        except:
            print('failed stack')
            return

        features_train = get_test_features(rec, stack)
        preds = self.regressor(features_train.to(y.device), x2.to(y.device))

        loss = quantile_loss(preds, y, self.quantiles)
        metrics = metric(
            preds * init_values.reshape(-1, 1).to(preds.device),
            y.to(preds.device) * init_values.to(preds.device))
        self.log('train_loss', loss)
        self.log('train_metric', metrics.mean())

        self.trainer.train_loop.running_loss.append(loss)

        opt = self.optimizers()

        old_weights = [self.regressor.state_dict()]
        if self.use_encoder:
            old_weights = [self.enc.state_dict()] + old_weights
        try:
            self.manual_backward(loss, opt)
            opt.step()
            opt.zero_grad()
            self.qtts.append(detached_rec)
        except:
            print('failed backprop')
            self.enc.load_state_dict(old_weights[0])
            self.regressor.load_state_dict(old_weights[1])
            self.log('failed back prop', 1)
            return {
                'preds': preds,
                'labels': y,
                'loss': loss}

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def on_train_epoch_start(self):
        print('Starting new epoch, num qtts: {}'.format(len(self.qtts)))
        self.qtts = []

    def validation_step(self, val_batch, batch_idx):
        if len(self.qtts) == 0:
            print('Empty stack')
            return

        x1, x2, y, init_values = val_batch
        accs = [None] * len(x1)
        for i, im in enumerate(x1):
            accs[i] = self.encoder(im=im)

        try:
            rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        except:
            print('val: failed qtt')
            return

        _, stack = get_train_features(
            merge_cores_to_batch(self.qtts), self.rank)

        features_val = get_test_features(rec, stack)
        preds = self.regressor(features_val.to(y.device), x2.to(y.device))

        loss = quantile_loss(preds, y, self.quantiles)
        metrics = metric(
            preds * init_values.reshape(-1, 1).to(y.device),
            y * init_values)
        self.log('val_loss', loss)
        self.log('val_metric', metrics.mean())

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=1e-3)


class TTRegressor(pl.LightningModule):
    def __init__(
            self,
            build_encoder_function: Callable,
            path_function: Callable,
            q_shape: Iterable[int],
            use_encoder: bool,
            ndims: int,
            rank: int,
            full_shape: Iterable[int],
            use_extra_features: bool):
        super().__init__()
        if use_encoder:
            build_enc_f = partial(
                build_encoder_function,
                arg_patch_func=add_dim,
                patch_func=path_function)
        else:
            build_enc_f=build_encoder_function

        self.qtts = []
        self.rank = rank
        self.q_shape = q_shape
        self.automatic_optimization = False
        self.use_encoder = use_encoder
        if use_encoder:
            self.enc = EncoderCA_3D(num_out_channels=ndims)
            self.encoder = partial(
                build_enc_f,
                enc=self.enc,
                q_shape=q_shape,
                full_shape=full_shape,
                detach=False,
                batch_size=10000)
        else:
            self.encoder = partial(
                build_enc_f,
                q_shape=q_shape,
                full_shape=full_shape)

        if use_extra_features:
            self.regressor = TwoHeadsRegressor(2, rank + 100, 3)
        else:
            self.regressor = TwoHeadsRegressor(0, rank, 3)

        self.quantiles = (0.2, 0.5, 0.8)

    def forward(self, x):
        if len(self.qtts) == 0:
            print('Empty stack')
            return

        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)

        rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        _, stack = get_train_features(
            merge_cores_to_batch(self.qtts), self.rank)
        features_val = get_test_features(rec, stack)

        return self.regressor(features_val.to(x.device), None)

    def training_step(self, batch, batch_idx):
        x, y = batch
        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)

        try:
            rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        except:
            import traceback
            traceback.print_exc()
            print('failed qtt')
            return

        detached_rec = tn.Tensor(
            [core.detach() for core in rec.cores], batch=True)

        try:
            _, stack = get_train_features(
                merge_cores_to_batch(self.qtts + [rec]), self.rank)
        except:
            import traceback
            traceback.print_exc()
            print('failed stack')
            return

        features_train = get_test_features(rec, stack)
        preds = self.regressor(features_train.to(y.device), None)

        loss = compute_MSE(y, preds[:, 1])
        self.log('train_loss', loss)

        metrics = compute_MSE(y, preds[:, 1])
        self.log('train_metric', metrics.mean())

        self.trainer.train_loop.running_loss.append(loss)

        opt = self.optimizers()

        self.old_weights = [self.regressor.state_dict(), self.optimizers().state_dict()]
        if self.use_encoder:
            self.old_weights = [self.enc.state_dict()] + self.old_weights
        try:
            self.manual_backward(loss, opt)
            opt.step()
            opt.zero_grad()

            for _, v in self.state_dict().items():
                if torch.isnan(v).any():
                    raise RuntimeError('nans in the weights')

            for _, v1 in self.optimizers().state_dict()['state'].items():
                for _, v2 in v1.items():
                    if isinstance(v2, torch.Tensor) and torch.isnan(v2).any():
                        raise RuntimeError('nans in the optimiser')
            self.qtts.append(detached_rec)
        except:
            import traceback
            traceback.print_exc()
            print('failed backprop')
            self.enc.load_state_dict(self.old_weights[0])
            self.regressor.load_state_dict(self.old_weights[1])
            self.optimizers().load_state_dict(self.old_weights[2])
            self.log('failed back prop', 1)
            return

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def on_train_epoch_start(self):
        print('Starting new epoch, num qtts: {}'.format(len(self.qtts)))
        self.qtts = []

    def validation_step(self, val_batch, batch_idx):
        if len(self.qtts) == 0:
            print('Empty stack')
            return

        x, y = val_batch
        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)

        try:
            rec, _ = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')
        except:
            print('failed qtt')
            return

        _, stack = get_train_features(
            merge_cores_to_batch(self.qtts), self.rank)

        features_val = get_test_features(rec, stack)
        preds = self.regressor(features_val.to(y.device), None)

        loss = compute_MSE(y, preds[:, 1])# quantile_loss(preds, y, self.quantiles)
        self.log('val_loss', loss)
        metrics = compute_MSE(y, preds[:, 1])
        self.log('val_metric', metrics.mean())

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=1e-3)


class ResNetQuantileRegressor(pl.LightningModule):
    def __init__(self, output_dim: int):
        super().__init__()
        self.resnet = generate_model(34, n_classes=0, n_input_channels=1)
        self.regressor = TwoHeadsRegressor(1, 512 + 100, output_dim).cuda()
        self.loss = compute_MSE
        self.metric = compute_MSE
        self.automatic_optimization = False
        self.quantiles = [0.2, 0.5, 0.8]

    def forward(self, x1, x2):
        return self.regressor(self.resnet(x1[:, None]), x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y, init_values = batch
        preds = self.regressor(self.resnet(x1[:, None]), x2)

        loss = quantile_loss(preds, y, self.quantiles)
        metrics = metric(
            preds * init_values.reshape(-1, 1).to(preds.device),
            y * init_values.to(preds.device))
        self.log('train_loss', loss)
        self.log('train_metric', metrics.mean())

        self.trainer.train_loop.running_loss.append(loss)

        opt = self.optimizers()
        self.manual_backward(loss, opt)
        opt.step()
        opt.zero_grad()

    def validation_step(self, val_batch, batch_idx):
        x1, x2, y, init_values = val_batch
        preds = self.regressor(self.resnet(x1[:, None]), x2)

        loss = quantile_loss(preds, y, self.quantiles)
        metrics = metric(
            preds * init_values.reshape(-1, 1).to(preds.device),
            y * init_values.to(preds.device))
        self.log('val_loss', loss)
        self.log('val_metric', metrics.mean())

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=1e-3)


class ResNetRegresor(pl.LightningModule):
    def __init__(self, output_dim: int):
        super().__init__()
        self.resnet = generate_model(34, n_classes=0, n_input_channels=1)
        self.regressor = Regressor(input_dim=512, output_dim=output_dim)
        self.loss = compute_MSE
        self.metric = compute_MSE
        self.automatic_optimization = False

    def forward(self, x):
        return self.regressor(self.resnet(x[:, None]))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss)

        metrics = self.metric(preds, y)
        self.log('train_metric', metrics)

        self.trainer.train_loop.running_loss.append(loss)

        opt = self.optimizers()

        try:
            self.manual_backward(loss, opt)
            opt.step()
            opt.zero_grad()

            return {
                'preds': preds,
                'labels': y,
                'loss': loss}
        except:
            print('Failed backprop, loss: {}'.format(loss))


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        preds = self.forward(x)

        loss = self.loss(preds, y)
        self.log('val_loss', loss)
        metric = self.metric(preds, y)
        self.log('val_metric', metric)

        return {
            'preds': preds,
            'labels': y,
            'loss': loss}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=1e-3)


class TTAE(pl.LightningModule):
    def __init__(
            self,
            build_encoder_function: Callable,
            patch_function: Callable,
            q_shape: Iterable[int],
            ndims: int,
            rank: int,
            full_shape: Iterable[int],
            arg_patch_function: Optional[Callable] = do_nothing):
        super().__init__()
        build_enc_f = partial(
            build_encoder_function,
            arg_patch_func=arg_patch_function,
            patch_func=patch_function,
            detach=False)
        self.automatic_optimization = False
        self.rank = rank
        self.q_shape = q_shape
        dims_enc = len(full_shape) - 1
        dims_dec = dims_enc + 1
        self.ndims = ndims
        self.enc = Encoder(num_neurons=[5**dims_enc, 256, 256, ndims], num_dims=dims_enc)
        self.dec = Decoder(num_neurons=[5**dims_dec, 256, 256, 1], num_dims=dims_dec)
        self.full_shape = full_shape
        self.encoder = partial(
            build_enc_f,
            enc=self.enc,
            q_shape=q_shape,
            full_shape=full_shape,
            batch_size=10000)

    def forward(self, x):
        accs = [None] * len(x)
        for i, im in enumerate(x):
            accs[i] = self.encoder(im=im)
        rec, infos = truncate_ca_qtt((accs, self.q_shape), self.rank, device='cpu')

        im_idxs = [None] * len(infos)
        for i, info in enumerate(infos):
            im_idxs[i] = np.hstack([
                np.ones((info['Xs'].shape[0], 1), dtype=np.int64) * i,
                np.vstack(
                    q_coords2full(
                        info['Xs'].detach().cpu().long().numpy(),
                        np.array(self.q_shape),
                        np.array(self.full_shape)))])
            im_idxs[i] = torch.tensor(im_idxs[i])

        im_idxs = torch.cat(im_idxs)

        random_idxs = torch.cat(
            [torch.tensor(np.random.choice(f_s, len(im_idxs))[None, ...]) for f_s in [len(infos)] + self.full_shape]).T
        im_idxs = torch.cat([im_idxs, random_idxs])

        patches = torch.cat(
            [idxmapper(
                rec[batch_el],
                im_idxs[im_idxs[:, 0] == batch_el][:, 1:].numpy(),
                self.full_shape,
                self.dec.rec_field)
            for batch_el in range(rec.shape[0])])

        ch = list(np.arange(len(patches.shape)))
        ch[1], ch[-1] = ch[-1], ch[1]
        sampled_output = self.dec(
            patches.permute(ch).to(x.device)).cpu()

        sampled_input = x[[np.zeros(len(im_idxs))] + [im_idxs[:, i] for i in range(im_idxs.shape[1] - 1)]].cpu()

        return sampled_output, sampled_input

    def training_step(self, batch, batch_idx):
        x, _ = batch
        sampled_preds, sampled_inpts = self.forward(x)
        loss = compute_MSE(sampled_preds, sampled_inpts)
        self.log('train_loss', loss)

        self.trainer.train_loop.running_loss.append(loss)

        opt = self.optimizers()
        self.manual_backward(loss, opt)
        opt.step()
        opt.zero_grad()

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        sampled_preds, sampled_inpts = self.forward(x)
        loss = compute_MSE(sampled_preds, sampled_inpts)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=1e-3)
