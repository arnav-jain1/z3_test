import numpy as np

x = 10
a = np.arange(5).reshape([5])
b = np.arange(5).reshape([5])

result = a * b
print(result.shape)