a = [1,2,3]
b = [2,3,4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product)

import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0


outputs = np.dot(weights, inputs) + bias

print(outputs)

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)

###Transpose###

a = [1,2,3]
a = np.array(a)
print(a,a.shape)

a = [1,2,3]
a = np.expand_dims(np.array(a), axis = 1)
print(a,a.shape)
print('Transposing......')
import numpy as np

a = [1,2,3]
b = [2,3,4]

a = np.array(a)
b = np.array(b).T
print(np.dot(a,b))
dot_product = np.dot(a,b)
print(dot_product)
######## There is a difference
import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T


print(np.dot(a, b))


