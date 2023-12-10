import torch

print(torch.__version__)

t1 = torch.tensor([1, 3, 2])
t2 = torch.tensor([2, 1, 2])

print(t1*t2)


for i in range(10):
    t1 += t2


print(t1)
