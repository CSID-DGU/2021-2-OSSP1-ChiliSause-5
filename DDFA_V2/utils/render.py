# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from ..Sim3DR import RenderPipeline
from .functions import plot_image
from .tddfa_util import _to_ctype

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res


#텍스처 렌더링
def render_with_texture_multiple(img, ver_lst, tri, tex_lst, show_flag=False, wfp=None):

    overlap = img.copy()

    for i, ver_ in enumerate(ver_lst):
        ver = _to_ctype(ver_.T)  # transpose
        print(ver)
        overlap = render_app(ver, tri, overlap, texture=tex_lst[i])

    res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res

def render_with_texture_single(img, ver_, tri, tex, show_flag=False, wfp=None, isBefore=True):
    overlap = img.copy()

    ver = _to_ctype(ver_.T)  # transpose
    overlap = render_app(ver, tri, overlap, texture=tex)

    res = overlap

    #이미지 자르기
    if isBefore:
        res = res[int(ver_[1].min())-10:int(ver_[1].max()+10)
                , int(ver_[0].min())-10:int(ver_[0].max())+10].copy()
    else:
        res = res[int(ver_[1].min()):int(ver_[1].max())
                , int(ver_[0].min()):int(ver_[0].max())].copy()

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res