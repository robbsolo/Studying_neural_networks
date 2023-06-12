import numpy as np

def relu(x):
	return np.maximum(0, x) # maximum used instead of max because it allows element-wise operations for matrices

def softmax(x):
	return np.exp(x - np.max(x))/np.sum(np.exp(x - np.max(x))) # x-np.max(x) is used to handle possible very large numbers; softmax is shift invariant

y = np.array([0, 0, 0, 1]) # AND operation; this is the target

epochs = 10000
batch_size = 1
features = 4
neurons = 5 # just one hidden layer
lr = 0.001

# initialize weights and biases randomly
x = np.random.randn(batch_size, features)
b1 = np.random.randn(neurons)
w1 = np.random.randn(features, neurons)
b2 = np.random.randn(features)
w2 = np.random.randn(neurons, features)

for epoch in range(epochs):

	# feedforward
	z1 = x.dot(w1)
	a1 = relu(z1 + b1)
	z2 = a1.dot(w2) # batch_size x features
	a2 = softmax(z2 + b2)

	# loss
	loss = float(-np.sum(y * np.log(a2))/features)
	if epoch % (epochs/10) == 0:
		print("Epoch: ", epoch, "; Loss: ", loss)
		for i in range(len(a2)):
			for j in range(len(a2[0])):
				a2[i][j] = round(a2[i][j], 2)
		print(a2) # print output

	# backpropagation
	da2 = (a2 - y)/features # batch_size x classes
	dw2 = a1.T.dot(da2) # batch_size x neurons dot batch_size x classes => a1 needs to be transposed
	db2 = np.sum(da2, axis=0)
	da1 = da2.dot(w2.T) # batch_size x neurons
	da1[z1 <= 0] = 0
	dw1 = x.T.dot(da1)
	db1 = np.sum(da1, axis=0)

	# gradient descent
	# the reason for subtraction is that the gradient returns the direction of steepest ascent,
	# so we subtract it
	w1 -= lr * dw1
	w2 -= lr * dw2
	b1 -= lr * db1
	b2 -= lr * db2
