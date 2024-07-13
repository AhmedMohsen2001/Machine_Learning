import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

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
    m=len(y)
    while epoch>=0:
        z=w.T@x
        loss=np.maximum(0,-y*z)
        mask=loss>0
        Loss=np.sum(loss)/m
        # print(f"Epoch: {Epoch-epoch} Loss: {Loss} w: {w.T}")
        if Loss*10**16<=tol:
            break
        dw=-((mask*y)@x.T).T
        w=w-r*dw
        epoch-=1
    print(f"Perc Epoch: {Epoch-epoch-1} Loss: {Loss}")
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
    m=len(y)
    while epoch>=0:
        z=w.T@x
        y_hat=1/(1+np.exp(-z))
        loss=-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)
        Loss=np.sum(loss)/m
        # print(f"Epoch: {Epoch-epoch} Loss: {Loss} w: {w.T}")
        if Loss<=tol:
            break
        dw=x@(y_hat-y).T
        w=w-r*dw
        epoch-=1
    print(f"LR Epoch: {Epoch-epoch-1} Loss: {Loss}")
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
    
def load_data(path,zeros=0,preproc=0,pn=0):
    x1=list()
    x2=list()
    y=list()
    proc=StandardScaler() 
    with open(path) as f:
        for line in f.readlines():
            line=line.split(',')
            x1.append(float(line[0]))
            x2.append(float(line[1]))
            y.append(float(line[2]))
    x=np.vstack([x1,x2]).T
    x_norm=proc.fit_transform(x)
    x=x.T if not zeros else np.vstack([x.T,np.ones((len(x1),))])
    x_norm=x_norm.T if not zeros else np.vstack([x_norm.T,np.ones((len(x1),))]) # m*n
    y=np.array(y).reshape(len(y),)
    if pn: y=(y-0.5)*2
    if not preproc: return x.T, y.astype(int)
    else: return x_norm.T, y.astype(int)
    
def plot_data(x,w,y,b=None,transpose=0,blocking=False):
    if not transpose: x=x.T
    w=w.reshape(-1,)
    x_0_start=np.min(x[0])
    x_0_end=np.max(x[0])
    x_1_start=-(w[0]*x_0_start+w[2])/w[1] if b==None else -(w[0]*x_0_start+b[0])/w[1]
    x_1_end=-(w[0]*x_0_end+w[2])/w[1] if b==None else -(w[0]*x_0_end+b[0])/w[1]
    p1=[x_0_start,x_0_end]
    p2=[x_1_start,x_1_end]
    pos_points = x[:, y == 1]
    neg_points = x[:, y == 0]
    plt.scatter(pos_points[0],pos_points[1],marker='x',c='r')
    plt.scatter(neg_points[0],neg_points[1],marker='x',c='g')
    plt.plot(p1,p2,c='b')
    plt.show(block=blocking)

def Logistic_Regression():
    path="ex2data1.txt"
    # Scikit
    x_scikit, y_scikit=load_data(path,0,1) # x: m*n
    log_reg=LogisticRegression(max_iter=10000, tol=0.01)
    log_reg.fit(x_scikit, y_scikit)
    w_scikit=log_reg.coef_.reshape(-1,)
    b_scikit=log_reg.intercept_
    y_hat_scikit=log_reg.predict(x_scikit).reshape(-1,).astype(int)
    loss=log_loss(y_scikit, y_hat_scikit)
    print(f"Scikit Epoch: {log_reg.n_iter_[0]} Loss: {loss}")
    plot_data(x_scikit,w_scikit,y_scikit,b_scikit,0,1)
    # Perceptron
    x_perc, y_perc=load_data(path,1,1,1)
    x_perc=x_perc.T
    w_perc=np.random.rand(x_perc.shape[0],)
    w_perc=perceptron_train(x_perc, y_perc, w_perc, 0.1, 100000, 0.01).reshape(-1,)
    z_perc=w_perc@x_perc
    y_perc=(y_perc/2+0.5).astype(int)
    y_hat_perc=(np.sign(z_perc).reshape(-1,)/2+0.5).astype(int)
    plot_data(x_perc,w_perc,y_perc,None,1,1)
    # Logistic Regression
    x_lr, y_lr=load_data(path,1,1)
    x_lr=x_lr.T 
    w_lr=np.random.rand(x_lr.shape[0],)
    w_lr=logistic_regression_train(x_lr, y_lr, w_lr, 0.1, 100000, 0.01).reshape(-1,)
    z_lr=w_lr@x_lr
    y_hat_lr=((1/(1+np.exp(-z_lr)))>=0.5).astype(int).reshape(-1,)
    plot_data(x_lr,w_lr,y_lr,None,1,1)
    # Print
    # a=30
    # slize=slice(0+a,15+a)
    # print(f"w_scikit: {w_scikit}  b_scikit: {b_scikit}  w_perc: {w_perc}  w_lr: {w_lr}")
    # print(f"y_hat_scikit:{y_hat_scikit[slize]}  y_hat_perc:{y_hat_perc[slize]}  y_hat_lr:{y_hat_lr[slize]}")
    # print(f"y:           {y_scikit[slize]}             {y_perc[slize]}           {y_lr[slize]}")
        
# perceptron()
# logistic_regression()
# scikit_LR()
Logistic_Regression()