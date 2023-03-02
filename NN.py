import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import time

params_exist = False
model_list=os.listdir("models")
if len(model_list)!=0:
    for i in range (1,len(model_list)+1):
        print(f"{i}. {model_list[i-1]}")
    model_num=int(input("Enter the which model to load: "))
    file_name=model_list[model_num-1]
    with open(file_name, "rb") as f:
        parameters = pickle.load(f)
    params_exist = True
    print(f"Parameters loaded from {file_name}")


# Importing the dataset
data=pd.read_csv("digit-recognizer/train.csv")
print("Csv file imported")
data=np.array(data)
m,n=data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

#initializing random weight and bias
def params():
    print("Initializing random weights and bias")
    W1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5

    return W1,b1,W2,b2


def relu(z):
    return np.maximum(z,0)

def softmax(z):
    s=np.exp(z)/sum(np.exp(z))
    return s

#forward propagation
def forward(W1,b1,W2,b2,X):
    z1=W1.dot(X)+b1
    A1=relu(z1)
    z2=W2.dot(A1)+b2
    A2=softmax(z2)

    return z1,z2,A1,A2

#one hot encoding
def one_hot(y):
    hot_y=np.zeros((y.size,y.max()+1))
    hot_y[np.arange(y.size),y]=1
    hot_y=hot_y.T
    return hot_y

#deriative of relu function
def relu_derivative(z):
    return z>0  

#backwarda propagation
def backward(X,Y,z1,z2,A1,A2,W2):
    hot_y=one_hot(Y)
    dz2= A2-hot_y
    dw2=1/m*dz2.dot(A1.T)
    db2=1/m*np.sum(dz2)
    dz1 = W2.T.dot(dz2) * relu_derivative(z1)
    dw1=1/m*dz1.dot(X.T)
    db1=1/m*np.sum(dz1)

    return dw1,db1,dw2,db2

#updating weights and bias
def update(W1,b1,W2,b2,dw1,db1,dw2,db2,alpha):
    W1=W1-alpha*dw1
    b1=b1-alpha*db1
    W2=W2-alpha*dw2
    b2=b2-alpha*db2

    return W1,b1,W2,b2


def predict(A):
    return np.argmax(A,0)

def acc(predictions,Y):
    return np.mean(predictions==Y)


count=0
last_pred=0
def early_stopping():
    global count,last_pred
    if acc(predict,Y_dev)<0.95:
        return True
    else:
        print("Accuracy reached 95%..")
        return False
    if last_pred==round(predictions,3):
        count+=1
    else:
        count=0
    last_pred==round(predictions,3)

    if count==8:
        print("Early Stopping..")
        return False
    else:
        return True

#training the model using gradient descent algorithm
def NN_model(X,Y,X_dev,Y_dev,num_iterations,alpha,param: list=None):
    if not param :
        W1,b1,W2,b2=params()
    else:
        W1,b1,W2,b2=param
    for i in range(0,num_iterations):
        z1,z2,A1,A2=forward(W1,b1,W2,b2,X)
        dw1,db1,dw2,db2=backward(X,Y,z1,z2,A1,A2,W2)
        W1,b1,W2,b2=update(W1,b1,W2,b2,dw1,db1,dw2,db2,alpha)
        if i%10==0:
            predictions=predict(A2)
            print(f"Epoch{i} Train_Acc:{acc(predictions,Y)}",end=" ")
            z1,z2,A1,A2=forward(W1,b1,W2,b2,X_dev)
            predictions=predict(A2)
            print(f"Val_Accuracy: {acc(predictions,Y_dev)}")

        if not early_stopping():
            break

    z1,z2,A1,A2=forward(W1,b1,W2,b2,X_dev)
    predictions=predict(A2)
    print("Final Dev Accuracy: ",acc(predictions,Y_dev))
    return W1,b1,W2,b2


# if parameters are found then it is initialized with those parameters
# else it is initialized with random parameters
train=input("Do you want to train the model? (y/n): ")
if train=='y':
    print("Starting training...")
    t=time.time()
    if params_exist:
        W1,b1,W2,b2=NN_model(X_train,Y_train,X_dev,Y_dev,500,0.1,parameters)
    else:
        W1,b1,W2,b2=NN_model(X_train,Y_train,X_dev,Y_dev,500,0.1)

    print("Training time: ",time.time()-t)

    choice=input("Do you want to save the parameters? (y/n): ")
    if choice=='y':
        model_name=input("Enter the name of the model: ")
        with open("models/"+model_name+".pkl", "wb") as f:
            pickle.dump([W1, b1, W2, b2], f)

def make_predictions(X):
    global W1,b1, W2, b2
    _, _, _, A2 = forward(W1, b1, W2, b2,X)
    predictions = predict(A2)
    return predictions

def test_prediction(index):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None])
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

while True:
    num=input("Enter the index of images you want to test/n to exit : ")
    if num!="n":
        test_prediction(int(num))
    else:
        break