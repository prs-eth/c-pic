# Cherry-Picking Gradients: Learning Low-Rank Embeddings of Visual Data via Differentiable Cross-Approximation (ICCV2021)

This is the repository with code for C-PIC experiments

## Abstract
We propose an end-to-end trainable framework that processes large-scale visual data tensors by looking at a fraction of their entries only. Our method combines a neural network encoder with a tensor train decomposition to learn a low-rank latent encoding, coupled with cross-approximation (CA) to learn the representation through a subset of the original samples. CA is an adaptive sampling algorithm that is native to tensor decompositions and avoids working with the full high-resolution data explicitly. Instead, it actively selects local representative samples that we fetch out-of-core and on-demand. The required number of samples grows only logarithmically with the size of the input. Our implicit representation of the tensor in the network enables processing large grids that could not be otherwise tractable in their uncompressed form. The proposed approach is particularly useful for large-scale multidimensional grid data (e.g., 3D tomography), and for tasks that require context over a large receptive field (e.g., predicting the medical condition of entire organs).

## Usage
Install deps: `pip install -r requirements.txt`

Launching the code:
`cd scripts`

```
python brats --batch_size 21 --use_encoder True --root "/scratch3/MICCAI_BraTS2020_TrainingData" --num_epochs 100 --use_extra_features False --ndims 16 --ngpus 1 --rank 10 --num_workers 1
```


## Cite
```
@InProceedings{c-pic_2021,
    author    = {Usvyatsov, Mikhail and Makarova, Anastasia and Ballester-Ripoll, Rafael and Rakhuba, Maxim and Krause, Andreas and Schindler, Konrad},
    title     = {Cherry-Picking Gradients: Learning Low-Rank Embeddings of Visual Data via Differentiable Cross-Approximation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11426-11435}
}
```
