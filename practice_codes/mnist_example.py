import sys, os
sys.path.append(os.pardir)      # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle

from dataset.mnist import load_mnist
from PIL import Image
from simple_neural_network import softmax, sigmoid

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(f"x_train shape : {x_train.shape}")
    print(f"t_train shape : {t_train.shape}")
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

########
# Test
########
x, t = get_data()
network = init_network()

batch_size = 100        # Batch
accuracy_cnt = 0

############
# Not Batch
############
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)    # Get largest percentage element's index
#     if p == t[i]:
#         accuracy_cnt += 1

########
# Batch
########
for i in range(0, len(x), batch_size):
    x_batch = x[i: i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])


print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
