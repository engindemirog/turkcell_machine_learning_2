import numpy as np

#inputs
temperature = 10
humidity = 10

X = np.array([temperature,humidity])


#weights
weights = np.array([0.4,0.6])

bias = -20


output = np.dot(X,weights)+bias

print(output)

#activation function -->true-false-->sigmoid

def sigmoid(x):
    return 1/(1+np.exp(-x))


activated_output = sigmoid(output)

print("Nöronun aktivasyon sonucunda verdiği tepki çıktısı : ", activated_output)
