import numpy as np

file = np.load('./a266.npz')
# with open("./a266.npz", 'w+') as f:
# print(file['data'].shape)
a = file['data'][1, ...]
a[a > 1] = 0
file['data'][1, ...] = a
print(np.max(file['data']))
# print(np.max(file['data']))
# for key, value in file.items():
#     print(key)