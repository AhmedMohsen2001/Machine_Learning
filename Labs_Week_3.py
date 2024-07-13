import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

def perceptron_init():                                     
    x_1=np.array([0.5,1,1.5,3,2,1])
    x_2=np.array([1.5,1,0.5,0.5,2,2.5])
    x_0=np.ones((len(x_1),))
    x=np.vstack((x_1,x_2,x_0))
    y=np.array([-1,-1,-1,1,1,1])
    w=np.array([1,2,3]).reshape(3,1)
    return x, y, w

def perceptron_train(x, y, w, r, epoch, tol):
    Epoch=epoch
    while epoch>=0:
        z=w.T@x
        loss=np.maximum(0,-y*z)
        mask=loss>0
        Loss=np.sum(loss)
        # print(f"Epoch: {Epoch-epoch} Loss: {Loss} w: {w.T}")
        if Loss*10**16<=tol:
            break
        dw=-((mask*y)@x.T).T
        w=w-r*dw
        epoch-=1
    return w
        

def perceptron():
    x,y,w=perceptron_init()
    w=perceptron_train(x, y, w, 0.1, 10000, 0.01)
    z=w.T@x
    y_hat=np.sign(z).astype(int).reshape(-1,)
    print(f"y_hat: {y_hat} y: {y}")
    
def logistic_regression_init():
    x_1=np.array([0.5,1,1.5,3,2,1])
    x_2=np.array([1.5,1,0.5,0.5,2,2.5])
    x_0=np.ones((len(x_1),))
    x=np.vstack((x_1,x_2,x_0))
    y=np.array([0,0,0,1,1,1])
    w=np.array([1,2,3]).reshape(3,1)
    return x, y, w

def logistic_regression_train(x, y, w, r, epoch, tol):
    Epoch=epoch
    while epoch>=0:
        z=w.T@x
        y_hat=1/(1+np.exp(-z))
        loss=-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)
        Loss=np.sum(loss)
        # print(f"Epoch: {Epoch-epoch} Loss: {Loss} w: {w.T}")
        if Loss<=tol:
            break
        dw=x@(y_hat-y).T
        w=w-r*dw
        epoch-=1
    return w

def logistic_regression():
    x, y, w=logistic_regression_init()
    w=logistic_regression_train(x, y, w, 0.1, 100000, 0.01)
    z=w.T@x
    y_hat=((1/(1+np.exp(-z)))>=0.5).astype(int).reshape(-1,)
    print(f"y_hat: {y_hat} y: {y}")
    
def scikit_LR_init():
    x_1=np.array([0.5,1,1.5,3,2,1])
    x_2=np.array([1.5,1,0.5,0.5,2,2.5])
    x_0=np.ones((len(x_1),))
    # x=np.vstack((x_1,x_2,x_0))
    x=np.vstack((x_1,x_2))
    y=np.array([0,0,0,1,1,1])
    w=np.random.rand(len(x),1)
    return x, y, w

def scikit_LR():
    x, y, w = scikit_LR_init()
    log_reg=LogisticRegression(max_iter=10000)
    log_reg.fit(x.T, y)
    w=log_reg.coef_
    b=log_reg.intercept_
    z=w@x+b
    y_hat=((1/(1+np.exp(-z)))>=0.5).astype(int).reshape(-1,)
    y_hat_LR=log_reg.predict(x.T).reshape(-1,)
    print(f"y_hat: {y_hat} y_hat_LR: {y_hat_LR} y: {y}")
        
perceptron()
logistic_regression()
scikit_LR()