# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from nets.network import Network
from model.config import cfg
from .utils import Adapt2CaffeData, load_url, download_url
from .nasnet import nasnet, nas_cpu, nas_gpu


class mobilenas(Network):
	def __init__(self, key="gpu"):
		Network.__init__(self)
		self.mobilenet, self.urls = nasnet(pretrained=False, key=key)

		self._feat_stride = [16, ]
		self._feat_compress = [1. / float(self._feat_stride[0]), ]
		self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
		self._net_conv_channels = 320
		self._fc7_channels = 1280
		self._net_conv_channels = self.mobilenet.feature_mix_layer.in_channels
		self._fc7_channels = self.mobilenet.feature_mix_layer.in_channels * 4


	def init_weights(self):
		def normal_init(m, mean, stddev, truncated=False):
			"""
			weight initalizer: truncated normal and random normal.
			"""
			if m.__class__.__name__.find('Conv') == -1:
				return
			if truncated:
				m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
			else:
				m.weight.data.normal_(mean, stddev)
			if m.bias is not None: m.bias.data.zero_()

		self.mobilenet.apply(lambda m: normal_init(m, 0, 0.09, True))
		normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

	def _image_to_head(self):
		net_conv = self._layers['head'](self._image)
		self._act_summaries['conv'] = net_conv

		return net_conv

	def _head_to_tail(self, pool5):
		fc7 = self._layers['tail'](pool5)
		fc7 = fc7.mean(3).mean(2)
		return fc7

	def _init_head_tail(self):
		if not hasattr(self, "mobilenet"):
			self.mobilenet = nas_gpu()

		# Fix blocks
		assert (0 <= cfg.MOBILENET.FIXED_LAYERS <= 12)
		for m in list(self.mobilenet.children())[:cfg.MOBILENET.FIXED_LAYERS]:
			for p in m.parameters():
				p.requires_grad = False

		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad = False

		self.mobilenet.apply(set_bn_fix)

		# Add weight decay
		def l2_regularizer(m, wd, regu_depth):
			if m.__class__.__name__.find('Conv') != -1:
				if regu_depth or m.groups == 1:
					m.weight.weight_decay = wd
				else:
					m.weight.weight_decay = 0

		self.mobilenet.apply(lambda x: l2_regularizer(x, cfg.MOBILENET.WEIGHT_DECAY, cfg.MOBILENET.REGU_DEPTH))

		# Build mobilenet.
		# self._layers['head'] = nn.Sequential(*list(self.mobilenet.children())[:12])
		# self._layers['tail'] = nn.Sequential(*list(self.mobilenet.children())[12:])
		self._layers['head'] = nn.Sequential(*list(self.mobilenet.features.children())[:-1])
		self._layers['tail'] = nn.Sequential(*list(self.mobilenet.features.children())[-1:])

	def train(self, mode=True):
		# Override train so that the training mode is set as we want
		nn.Module.train(self, mode)
		if mode:
			# Set fixed blocks to be in eval mode (not really doing anything)
			for m in list(self.mobilenet.children())[:cfg.MOBILENET.FIXED_LAYERS]:
				m.eval()

			# Set batchnorm always in eval mode during training
			def set_bn_eval(m):
				classname = m.__class__.__name__
				if classname.find('BatchNorm') != -1:
					m.eval()

			self.mobilenet.apply(set_bn_eval)

	def load_pretrained_cnn_from_url(self, url=None):
		if url is None:
			self.mobilenet.load_state_dict(load_url(self.urls["weight"]), strict=False)
		else:
			Warning("Ensure your weight file is corresponding to net.config")
			self.mobilenet.load_state_dict(load_url(url["weight"]), strict=False)

	def load_pretrained_cnn(self, state_dict):
		DeprecationWarning("This API should NOT be called when using MobileNet V2")
		print('Warning: No available pretrained model yet')
		self.mobilenet.load_state_dict({k: state_dict['features.' + k] for k in list(self.mobilenet.state_dict())})
		# url = "http://file.lzhu.me/pytorch/models/mobilenet_v2-ecbe2b56.pth.tar"
		# fp = model_zoo.load_url(url, map_location="cpu")
		# self.mobilenet.load_state_dict(fp)
