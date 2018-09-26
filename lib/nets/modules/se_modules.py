import torch.nn as nn


class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel),
			nn.Sigmoid()
		)

	def forward(self, x):
		n, c, h, w = x.size()
		out = self.global_avg_pool(x).view(n, c) # squezze last two dimensions
		out = self.fc(out).view(n, c , 1, 1) # expand last two dimensions
		return x * out


def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3,  stride=stride, padding=1, bias=False)

# SE-ResNet
class SEBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
		super(SEBasicBlock, self).__init__()
		self.stride = stride

		self.conv1 = nn.Sequential(
			conv3x3(inplanes, planes, stride),
			nn.BatchNorm2d(planes),
			nn.ReLU(inplace=True),
		)

		self.conv2 = nn.Sequential(
			conv3x3(planes, planes, 1),
			nn.BatchNorm2d(planes),
			SELayer(planes, reduction)
		)
		self.downsample = downsample
		self.final_relu = nn.ReLU(inplace=True)


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.conv2(out)

		if self.downsample is not None:
			residual = self.downsample(residual)

		return self.final_relu(out + residual)


class SEBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
		super(SEBottleneck, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
			nn.BatchNorm2d(planes),
			nn.ReLU(inplace=True),
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(planes),
			nn.ReLU(inplace=True),
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
			nn.BatchNorm2d(planes * 4),
			SELayer(planes * 4, reduction)
		)

		self.downsample = downsample
		self.stride = stride
		self.final_relu = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		return self.final_relu(out + residual)


from torchvision.models import ResNet
def se_resnet34(num_classes):
	"""Constructs a ResNet-34 model.
	Args:
	    pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
	model.avgpool = nn.AdaptiveAvgPool2d(1)
	return model

if __name__ == '__main__':
	import torch
	net = se_resnet34(1000)
	data = torch.zeros(1, 3, 224, 224)
	out = net(data)
	print(out.size())