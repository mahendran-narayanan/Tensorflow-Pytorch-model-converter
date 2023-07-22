import torch
import tensorflow as tf
import re
from torch import nn

from model import *

def torch_converter():
	ls = model.layers
	order = []
	layers = []
	batchfilters=0
	for i in range(len(ls)):
		if re.search('conv2d',model.layers[i].name):
			filtersize = ls[i].filters
			kernel_size = ls[i].kernel_size
			batchfilters = ls[i].filters
			strides = ls[i].strides
			order.append('conv_'+str(filtersize)+'_'+str(kernel_size[0])+'_'+str(strides[0]))
			layers.append(torch.nn.Conv2d(filtersize,filtersize,kernel_size))
		elif re.search('batch_norm',model.layers[i].name):
			order.append('batchnorm_'+str(batchfilters))

	return order


class AlteredModel(nn.Module):
	def __init__(self,total_layers):
		super().__init__()
		self.layers = self.model_maker(total_layers)

	def model_maker(self,layers):
		tmodel = []
		for i in range(len(layers)):
			tre = layers[i].split('_')
			if tre[0]=='conv':
				tmodel.append(torch.nn.Conv2d(int(tre[1]),int(tre[1]),int(tre[2]),stride=int(tre[3])))
			elif tre[0]=='batchnorm':
				tmodel.append(torch.nn.BatchNorm2d(int(tre[1])))
		return torch.nn.Sequential(*tmodel)

	def forward(self,x):
		return self.layers(x)

total_layers = torch_converter()
model = AlteredModel(total_layers)
print(model)

