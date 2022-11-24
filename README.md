# Warper

This is an implementation of the waping process presented in Chapter 3 in the
paper "DocUNet: Document Image Unwarping via a Stacked U-Net".

```
@InProceedings{Ma_2018_CVPR,
author = {Ma, Ke and Shu, Zhixin and Bai, Xue and Wang, Jue and Samaras, Dimitris},
title = {DocUNet: Document Image Unwarping via a Stacked U-Net},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

This code was produced with the help of:
- https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
- https://github.com/mhashas/Document-Image-Unwarping-pytorch/issues/7

This code aims for a better variable naming and better structure.

## Benchmarking

With the current implementation, the `hyperfine` command shows the following
times for an image with dimensions (1776, 1200):

```
Time (mean ± σ):      3.904 s ±  1.948 s    [User: 3.541 s, System: 1.017 s]
Range (min … max):    1.157 s …  6.252 s    10 runs
```