import math

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from models.modules.layers import *


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride, ops_order):
	if candidate_ops is None:
		""" ops set used in `https://arxiv.org/abs/1806.02639` """
		candidate_ops = [
			'1x1_Conv', '3x3_DepthConv', '5x5_DepthConv', '7x7_DepthConv', 'Identity', '3x3_MaxPool', '3x3_AvgPool',
		]

	name2ops = {
		'1x1_Conv': lambda in_C, out_C, S: ConvLayer(in_C, out_C, 1, stride=S, ops_order=ops_order),
		'3x3_Conv': lambda in_C, out_C, S: ConvLayer(in_C, out_C, 3, stride=S, ops_order=ops_order),
		'3x3_DepthConv': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 3, stride=S, ops_order=ops_order),
		'5x5_DepthConv': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 5, stride=S, ops_order=ops_order),
		'7x7_DepthConv': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 7, stride=S, ops_order=ops_order),
		'3x3_DilConv': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 3, stride=S, dilation=2, ops_order=ops_order),
		'5x5_DilConv': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 5, stride=S, dilation=2, ops_order=ops_order),
		'1x3_3x1_Conv': lambda in_C, out_C, S: VHConvLayer(in_C, out_C, 3, stride=S, ops_order=ops_order),
		'1x5_5x1_Conv': lambda in_C, out_C, S: VHConvLayer(in_C, out_C, 5, stride=S, ops_order=ops_order),
		'1x7_7x1_Conv': lambda in_C, out_C, S: VHConvLayer(in_C, out_C, 7, stride=S, ops_order=ops_order),
		'Identity': lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
		'3x3_MaxPool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'max', 3, stride=S, ops_order=ops_order),
		'3x3_AvgPool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'avg', 3, stride=S, ops_order=ops_order),
		'5x5_MaxPool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'max', 5, stride=S, ops_order=ops_order),
		'7x7_MaxPool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'max', 7, stride=S, ops_order=ops_order),
		'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
		'3x3_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1, False),
		'3x3_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3, False),
		'3x3_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6, False),
		'4x4_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 4, S, 1, False),
		'4x4_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 4, S, 3, False),
		'4x4_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 4, S, 6, False),
		'5x5_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1, False),
		'5x5_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3, False),
		'5x5_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6, False),
		'6x6_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 6, S, 1, False),
		'6x6_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 6, S, 3, False),
		'6x6_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 6, S, 6, False),
		'7x7_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1, False),
		'7x7_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3, False),
		'7x7_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6, False),
		'3x3_MBConv1_RELU': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1, True),
		'4x4_MBConv1_RELU': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 4, S, 1, True),
		'5x5_MBConv1_RELU': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1, True),
		'6x6_MBConv1_RELU': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 6, S, 1, True),
		'7x7_MBConv1_RELU': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1, True),
		'MaxPool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'max', S, stride=S, ops_order=ops_order),
		'AvgPool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'avg', S, stride=S, ops_order=ops_order),
	}

	return [
		name2ops[name](in_channels, out_channels, stride) for name in candidate_ops
	]


class RealRedundantEdge(BasicUnit):

	def __init__(self, candidate_ops, AP_path_alpha=None):
		super(RealRedundantEdge, self).__init__()

		self.candidate_ops = nn.ModuleList(candidate_ops)
		if AP_path_alpha is None:
			self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))
			self.own_arch_param = True
		else:
			self.AP_path_alpha = AP_path_alpha
			self.own_arch_param = False

		self.register_buffer('_mask', torch.ones(self.n_choices))

	@property
	def n_choices(self):
		return len(self.candidate_ops)

	@property
	def mask(self):
		return Variable(self._mask)

	@property
	def probs_over_ops(self):
		probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
		probs = probs * self.mask  # mask
		probs = probs / torch.sum(probs)  # rescale to 1
		return probs

	@property
	def chosen_index(self):
		probs = self.probs_over_ops.data.cpu().numpy()
		index = int(np.argmax(probs))
		return index, probs[index]

	@property
	def chosen_op(self):
		index, _ = self.chosen_index
		return self.candidate_ops[index]

	@property
	def random_op(self):
		index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
		return self.candidate_ops[index]

	def forward(self, x):
		outs = []
		probs = self.probs_over_ops
		for _i in range(self.n_choices):
			if probs.data[_i] > 0:
				outs.append(
					probs[_i] * self.candidate_ops[_i](x)
				)
			else:
				outs.append(0)
		return list_sum(outs)

	@property
	def unit_str(self):
		chosen_index, probs = self.chosen_index
		return 'R(%s, %.3f)' % (self.candidate_ops[chosen_index].unit_str, probs)

	@property
	def config(self):
		raise ValueError('not needed')

	@staticmethod
	def build_from_config(config: object) -> object:
		raise ValueError('not needed')

	def get_flops(self, x):
		return self.chosen_op.get_flops(x)

	@property
	def entropy_loss(self):
		""" flat -> large entropy, regularization """
		if self.own_arch_param:
			eps = 1e-8
			probs = self.probs_over_ops
			log_probs = torch.log(probs + eps)
			entropy = - torch.sum(torch.mul(probs, log_probs))
		else:
			entropy = 0
		return entropy

	@property
	def n_remaining(self):
		remain_num = np.sum(self._mask.cpu().numpy())
		return remain_num

	def prune_unimportant_path(self):
		probs = self.probs_over_ops.data.cpu().numpy()
		min_val, min_idx = 1, 0
		for _i in range(0, self.n_choices):
			if 0 < probs[_i] < min_val:
				min_val = probs[_i]
				min_idx = _i

		assert self.n_remaining > 1, 'you are masking the last candidate op'
		# mask the chosen index
		self._mask[min_idx] = 0


class BinaryRedundantEdge(RealRedundantEdge):
	mode = None

	def __init__(self, candidate_ops, AP_path_alpha=None, AP_path_wb=None, sample_single_op=True):
		super(BinaryRedundantEdge, self).__init__(candidate_ops, AP_path_alpha)

		if AP_path_wb is None:
			self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))
		else:
			self.AP_path_wb = AP_path_wb
		self.sample_single_op = sample_single_op

		self._active_index = None
		self._inactive_index = None
		self._log_prob = None

	def binarize(self):
		if self.own_arch_param:
			self._log_prob = None
			self.AP_path_wb.data.zero_()
			# binarize according to probs
			probs = self.probs_over_ops
			if BinaryRedundantEdge.mode == 'force_two':
				# sample two ops according to `probs`
				sample_op = torch.multinomial(probs.data, 2, replacement=False)
				probs_slice = F.softmax(torch.cat([
					self.AP_path_alpha[idx] for idx in sample_op
				]), dim=0)
				if self.sample_single_op:
					c = torch.multinomial(probs_slice.data, 1)[0]
					sample = sample_op[c]
					self.AP_path_wb.data[sample] = 1.0
					self._active_index = [sample]
					self._inactive_index = [sample_op[1 - c]]
				else:
					c = torch.bernoulli(probs_slice.data)
					for _i, idx in enumerate(sample_op):
						self.AP_path_wb.data[idx] = c[_i]
						if c[_i] > 0:
							self._active_index.append(_i)
						else:
							self._inactive_index.append(_i)
			else:
				if self.sample_single_op:
					sample = torch.multinomial(probs.data, 1)[0]
					self.AP_path_wb.data[sample] = 1.0
					self._active_index = [sample]
					self._inactive_index = [_i for _i in range(0, sample)] + \
					                       [_i for _i in range(sample + 1, self.n_choices)]
					self._log_prob = torch.log(probs[sample])
				else:
					self._active_index = []
					self._inactive_index = []
					self._log_prob = 0
					sample = torch.bernoulli(probs.data)
					self.AP_path_wb.data.copy_(sample)
					for _i in range(self.n_choices):
						if sample[_i] > 0:
							self._active_index.append(_i)
							self._log_prob = self._log_prob + torch.log(probs[_i])
						else:
							self._inactive_index.append(_i)
							self._log_prob = self._log_prob + torch.log(1 - probs[_i])
			# avoid over-regularization as suggested in 'http://proceedings.mlr.press/v80/bender18a.html'
			for _i in range(self.n_choices):
				for name, param in self.candidate_ops[_i].named_parameters():
					param.grad = None

	@property
	def active_index(self):
		# generate index with nonzero binary value
		for _i in self._active_index:
			yield _i

	@property
	def inactive_index(self):
		for _i in self._inactive_index:
			yield _i

	def forward(self, x):
		if BinaryRedundantEdge.mode is None:
			return self.candidate_ops[self._active_index[0]](x)
		# output = 0
		# for _i in self.active_index:
		#	output = output + self.AP_path_wb[_i] * self.candidate_ops[_i](x)
		# if len(self._active_index) == 0:
		#	batch_size, _, h, w = x.size()
		#	out_channels = self.candidate_ops[0].out_channels
		#	stride = self.candidate_ops[0].__dict__.get('stride', 1)
		#	if x.is_cuda:
		#		with torch.cuda.device(x.get_device()):
		#			padding = torch.cuda.FloatTensor(batch_size, out_channels, h // stride, w // stride).fill_(0)
		#	else:
		#		padding = torch.zeros(batch_size, out_channels, h // stride, w // stride)
		#	output = torch.autograd.Variable(padding, requires_grad=False)
		# return output
		elif BinaryRedundantEdge.mode == 'force_full':
			output = 0
			for _i in self.active_index:
				oi = self.candidate_ops[_i](x)
				output = output + self.AP_path_wb[_i] * oi
			for _i in self.inactive_index:
				oi = self.candidate_ops[_i](x)
				output = output + self.AP_path_wb[_i] * oi.detach()
			return output
		elif BinaryRedundantEdge.mode == 'force_two':
			BinaryRedundantEdge.mode = 'force_full'
			output = self.forward(x)
			BinaryRedundantEdge.mode = 'force_two'
			return output
		else:
			raise NotImplementedError

	@property
	def unit_str(self):
		chosen_index, probs = self.chosen_index
		return 'B(%s, %.3f)' % (self.candidate_ops[chosen_index].unit_str, probs)

	@property
	def config(self):
		raise ValueError('not needed')

	@staticmethod
	def build_from_config(config: object) -> object:
		raise ValueError('not needed')

	def get_flops(self, x):
		flops = 0
		for i in self.active_index:
			delta_flop, _ = self.candidate_ops[i].get_flops(x)
			flops += delta_flop
		return flops, self.forward(x)

	def set_arch_param_grad(self):
		if self.own_arch_param:
			binary_grads = self.AP_path_wb.grad.data
			if self.AP_path_alpha.grad is None:
				self.AP_path_alpha.grad = Variable(torch.zeros(self.n_choices).cuda(), volatile=True)
			if BinaryRedundantEdge.mode == 'force_two':
				involved_idx = self._active_index + self._inactive_index
				probs_slice = F.softmax(torch.cat([
					self.AP_path_alpha[idx] for idx in involved_idx
				]), dim=0).data
				for i in range(2):
					for j in range(2):
						origin_i = involved_idx[i]
						origin_j = involved_idx[j]
						self.AP_path_alpha.grad.data[origin_i] += \
							binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
				for _i, idx in enumerate(self._active_index):
					self._active_index[_i] = (idx, self.AP_path_alpha.data[idx])
				for _i, idx in enumerate(self._inactive_index):
					self._inactive_index[_i] = (idx, self.AP_path_alpha.data[idx])
			else:
				probs = self.probs_over_ops.data
				for i in range(self.n_choices):
					for j in range(self.n_choices):
						if binary_grads[j] == 0:
							continue
						self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])

	def balance_for_force_two(self):
		involved_idx = [idx for idx, _ in (self._active_index + self._inactive_index)]
		old_alphas = [alpha for _, alpha in (self._active_index + self._inactive_index)]
		new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

		offset = math.log(
			sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
		)

		for idx in involved_idx:
			self.AP_path_alpha.data[idx] -= offset

	@property
	def log_prob(self):
		if self.own_arch_param:
			return self._log_prob
		else:
			return 0

	def set_chosen_op_active(self):
		chosen_idx, _ = self.chosen_index
		self._active_index = [chosen_idx]
		self._inactive_index = [_i for _i in range(0, chosen_idx)] + \
		                       [_i for _i in range(chosen_idx + 1, self.n_choices)]

	def is_zero_layer(self):
		return self.candidate_ops[self._active_index[0]].is_zero_layer()
