from mnist import MNIST
import sys
sys.path.append('../')

import torchvision
from models.lightning import TTAE, add_dim, do_nothing
from torch.utils.data import Subset, DataLoader
from utils.accessors import build_enc_acc2d_multidim
from utils.helpers_2d import get_imgs_patches
import pytorch_lightning as pl
import torch

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0, 1)])
ds = torchvision.datasets.MNIST(root='../../sandbox_data/', transform=trans)
train_dataset = Subset(ds, [0])
val_dataset = Subset(ds, [0])

train_dl = DataLoader(train_dataset, batch_size=1)
val_dl = DataLoader(val_dataset, batch_size=1)

ttae = TTAE(
    build_encoder_function=build_enc_acc2d_multidim,
    patch_function=get_imgs_patches,
    arg_patch_function=do_nothing,
    q_shape=[2]*10 + [2, 2, 2],
    ndims=8,
    rank=10,
    full_shape=[32, 32, 8])

trainer = pl.Trainer(   
            max_epochs=1000,
            gpus=1,
#             logger=tb_logger,
#             resume_from_checkpoint=pretrain,
            num_processes=1)
trainer.fit(ttae, train_dl, val_dl)

