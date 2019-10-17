# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:28:23 2019

@author: Admin
"""
import torch
import torch.nn.functional as F

x = torch.randn(5, 3)
print(x)
x0 = F.softmax(x, dim=1)
print(x0)
output=torch.argmax(x0,1)

for i in x0:
    print(torch.argmax(i))
print(output)
#print(x0.sum(0))

#temp=torch.randn(1,3)