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


def conv_bn(inp, oup, stride):
	return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
	return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(  # dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),  # pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), )
		else:
			self.conv = nn.Sequential(  # pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),  # pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), )

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


class MobileNetV2(nn.Module):
	def __init__(self, n_class=1000, input_size=224, width_mult=1.):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [  # t, c, n, s
			[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1], ]

		# building first layer
		assert input_size % 32 == 0
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn(3, input_channel, 2)]
		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
				else:
					self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
				input_channel = output_channel
		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, n_class), )

		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		print(x.size())
		x = x.mean(3).mean(2)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


def mnet_v2(pretrained=False):
	model = MobileNetV2()
	if pretrained:
		url = "http://file.lzhu.me/pytorch/models/mobilenet_v2-ecbe2b56.pth.tar"
		fp = model_zoo.load_url(url, map_location="cpu")
		model.load_state_dict(fp)
	return model


class mobilenetv2(Network):
	def __init__(self):
		Network.__init__(self)
		self._feat_stride = [16, ]
		self._feat_compress = [1. / float(self._feat_stride[0]), ]
		self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
		self._net_conv_channels = 320
		self._fc7_channels = 1280

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
		self.mobilenet = mnet_v2()

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

	def load_pretrained_cnn_from_url(self, url="http://file.lzhu.me/pytorch/models/mobilenet_v2-ecbe2b56.pth.tar"):
		fp = model_zoo.load_url(url, map_location="cpu")
		self.mobilenet.load_state_dict(fp)

	def load_pretrained_cnn(self, state_dict):
		Warning("This API should NOT be called when using MobileNet V2")
		print('Warning: No available pretrained model yet')
		self.mobilenet.load_state_dict({k: state_dict['features.' + k] for k in list(self.mobilenet.state_dict())})
		# url = "http://file.lzhu.me/pytorch/models/mobilenet_v2-ecbe2b56.pth.tar"
		# fp = model_zoo.load_url(url, map_location="cpu")
		# self.mobilenet.load_state_dict(fp)
