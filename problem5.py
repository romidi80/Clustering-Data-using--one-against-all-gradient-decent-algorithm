import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def square_loss(y_pred, target):
    return np.mean(np.power((y_pred-target), 2))

def one_hot(special_one,y):
    onehot = [None]*len(y)
    i = 0
    for label in y:
        if label == special_one:
            onehot[i] = 1        
        else:
            onehot[i] = 0
        i +=1
    return onehot

def train_and_test_split(df):
    train_data = df.sample(frac = 0.8)
    test_data = df.drop(train_data.index)
    y_train = train_data.iloc[:,-1]
    y_test = test_data.iloc[:,-1]
    train_data = train_data.iloc[:,:-1]
    test_data = test_data.iloc[:,:-1]
    X_train = train_data.iloc[: , :-1]
    X_test = train_data.iloc[: , :-1]
    return X_test, X_train, y_test, y_train

def gradiant_descent(X_train,y_train,sepration):
    lr = 0.01 
    W = np.random.uniform(0,1000,(2,1))

    y_train = one_hot(sepration,y_train)
    for i in range(5000):
        z = np.dot(X_train, W)
        y_pred = sigmoid(z)
        l = square_loss(y_pred, y_train)
        gradient_W = np.array(np.dot((y_pred.reshape(len(y_pred,)) - y_train).T, X_train))/X_train.shape[0]
        gradient_W = gradient_W.reshape(2,1)
        W = W - lr * gradient_W
        if i%100 == 0:
            print(i, "epocs passed!")
    return z

data = pd.read_csv("Stress-Lysis.csv")
df = pd.DataFrame(data)
x = df.iloc[: , :-1]
y = df.iloc[: , -1]
colors = {'high':'orange', 'mid':'blue', 'low':'green'}

double_feat = []
for feat1 in x:
    for feat2 in x:
        s = set()
        s.add(feat1)
        s.add(feat2)
        if s not in double_feat and feat1 != feat2:
           double_feat.append(s)
           fig, ax = plt.subplots()
           ax.scatter(x[feat1],x[feat2], c=y.map(colors))
           ax.set_xlabel(feat1)
           ax.set_ylabel(feat2)

X_test, X_train, y_test, y_train = train_and_test_split(df)

y_pred = gradiant_descent(X_train,y_train,"high")    
print("first sepration done!")

fig, ax = plt.subplots()
ax.scatter(X_train["Humidity"],X_train["Temperature"])
ax.plot(X_train["Humidity"], y_pred)
ax.set_xlabel("Humidity")
ax.set_ylabel("Temperature")
ax.set_ybound(77.5, 100)
plt.show()
print((1-sum(sum(y_pred - y_train)))*100)



       