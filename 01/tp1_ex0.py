from keras.datasets import mnist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import np_utils

def accuracy(W, b, images, labels):
  pred = forward(images, W,b )
  return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0

def softmax(X):
 # Input matrix X of size Nbxd - Output matrix of same size
 E = np.exp(X)
 return (E.T / np.sum(E,axis=1)).T

def forward(batch,W,b):
  S = np.matmul(batch,W) + b
  Y = softmax(S)

  return Y

def backward(N,batch,W,b,Y,Yc,eta):
  nabla_W = (1/N)*np.matmul(batch.T,(Y-Yc))
  nabla_b = (1/N)*np.sum(((Y-Yc)))

  W_n = W - eta*nabla_W
  b_n = b - eta*nabla_b
  return W_n,b_n

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

K=10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

mpl.use('TKAgg')
plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
  plt.subplot(10,20,i+1)
  plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
  plt.axis('off')


N = X_train.shape[0]
d = X_train.shape[1]
W = np.random.rand(d,K)
b = np.random.rand(1,K)
numEp = 20 # Number of epochs for gradient descent
eta = 1e-1 # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))


batches = np.split(X_train,nb_batches)
batches_y = np.split(Y_train,nb_batches)

acc_t = 0
intera =0 

while acc_t<92:
  for epoch in range(numEp):
    for ex in range(nb_batches):
          Y = forward(batches[ex], W, b)
          W,b = backward(batch_size,batches[ex],W,b,Y,batches_y[ex],eta)
          acc = accuracy(W,b,batches[ex],batches_y[ex])
          #print(acc)
    acc_t = accuracy(W,b,X_test,Y_test)
    if(acc_t>92):
      break
    intera = intera + 1       

print(acc_t,intera)

plt.show()