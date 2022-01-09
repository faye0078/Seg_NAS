# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from model.deeplabv3plus.deeplabv3plus import deeplabv3plus
from model.deeplabv3plus.config import Configuration

def generate_net(args):
	if args.model_name == 'deeplabv3plus' or args.model_name == 'deeplabv3+':
		config = Configuration()
		return deeplabv3plus(config)

	else:
		raise ValueError('generateNet.py: network %s is not support yet'%args.model_name)
