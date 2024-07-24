import torch
x = torch.rand(5, 3)
print(torch.cuda.is_available())
print (torch.cuda.get_device_name(0))
import numpy as np
import matplotlib.pyplot as plt
print(x)