# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np
import os.path as osp

from ..Sim3DR import rasterize
from .functions import plot_image
from .io import _load, _dump
from .tddfa_util import _to_ctype

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def calc_ncc_code():
    from bfm import bfm

    # formula: ncc_d = ( u_d - min(u_d) ) / ( max(u_d) - min(u_d) ), d = {r, g, b}
    u = bfm.u
    u = u.reshape(3, -1, order='F')

    for i in range(3):
        u[i] = (u[i] - u[i].min()) / (u[i].max() - u[i].min())

    _dump('../configs/ncc_code.npy', u)


def pncc(img, ver_lst, tri, show_flag=False, wfp=None, with_bg_flag=True):
    ncc_code = _load(make_abs_path('../configs/ncc_code.npy'))

    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    # rendering pncc
    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = rasterize(ver, tri, ncc_code.T, bg=overlap)  # m x 3

    if wfp is not None:
        cv2.imwrite(wfp, overlap)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(overlap)

    return overlap


def main():
    # `configs/ncc_code.npy` is generated by `calc_nnc_code` function
    # calc_ncc_code()
    pass


if __name__ == '__main__':
    main()
