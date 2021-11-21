# coding: utf-8

__author__ = 'cleardusk'

import sys
import os.path as osp

sys.path.append('..')

import torch
from .. import models
from .tddfa_util import load_model

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)

def convert_to_onnx(**kvs):
    # 1. load model
    size = kvs.get('size', 120)
    model = getattr(models, kvs.get('arch'))(
        num_classes=kvs.get('num_params', 62),
        widen_factor=kvs.get('widen_factor', 1),
        size=size,
        mode=kvs.get('mode', 'small')
    )
    checkpoint_fp = make_abs_path('../'+kvs.get('checkpoint_fp'))
    model = load_model(model, checkpoint_fp)
    model.eval()

    # 2. convert
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, size, size)
    wfp = checkpoint_fp.replace('.pth', '.onnx')
    torch.onnx.export(
        model,
        (dummy_input, ),
        wfp,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True
    )
    print(f'Convert {checkpoint_fp} to {wfp} done.')
    return wfp
