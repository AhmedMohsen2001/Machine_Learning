import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

def load_data(path):
    x0=list()
    x1=list()
    x2=list()
    x3=list()
    y=list()
    with open(path) as f:
        for line in f.readlines():
            line=line.split(',')
            x0.append(float(line[0]))
            x1.append(float(line[1]))
            x2.append(float(line[2]))
            x3.append(float(line[3]))
            y.append(float(line[4]))
    f.close()
    return np.array(x0), np.array(x1), np.array(x2), np.array(x3), np.array(y).reshape(len(y),1)

def train(x, y, w, b, r=0.01, epoch=100, tol=0.01):
    m=len(y)
    Epoch=epoch
    w_new=w.copy()
    while epoch>=0:
        y_pred=x@w.T+b
        loss=1/(2*m)*np.sum(((y_pred-y))**2)
        print(f"Epoch: {Epoch-epoch}, Loss: {loss}, w: {w}, b: {b}")
        if loss<tol:
            break  
        dw=(1/m)*(x.T@(y_pred-y))
        db=(1/m)*np.sum(y_pred-y)
        w=(w.T-r*dw).T
        b=b-r*db
        epoch-=1
    return w, b

def weights_init(w0, w1, w2, w3, b):
    return np.array([w0,w1,w2,w3]).reshape(1,4), b   

def data_init(x0,x1,x2,x3):
    x_train_0=np.array(x0)#.reshape(3,1)
    x_train_1=np.array(x1)#.reshape(3,1)
    x_train_2=np.array(x2)#.reshape(3,1)
    x_train_3=np.array(x3)#.reshape(3,1)
    x_train=np.array([x0, x1, x2, x3]).T
    mean=np.mean(x_train,axis=0)
    std=np.std(x_train,axis=0)
    x_norm=(x_train-mean)/std
    return x_train, x_norm, mean, std

def get_data(mean, std):
    x_in=list()
    for i in range(4):
        x_in.append((float(input(f"Enter the value of x{i}: "))-mean[i])/std[i])
    return np.array(x_in).reshape(1,4)

def gd_handmade():
    path='D:\Engineer\Machine Learning Andrew NG\Machine-Learning-Specialization-Coursera\C1 - Supervised Machine Learning - Regression and Classification\week2\Optional Labs\data\houses.txt'
    x_train_0, x_train_1, x_train_2, x_train_3, y_train=load_data(path)
    # x_train_0=np.array([952, 1244, 1947])#.reshape(3,1)
    # x_train_1=np.array([2, 3, 3])#.reshape(3,1)
    # x_train_2=np.array([1, 2, 2])#.reshape(3,1)
    # x_train_3=np.array([65, 64, 17])#.reshape(3,1)
    x_train, x_norm, mean, std=data_init(x_train_0,x_train_1,x_train_2,x_train_3)
    # y_train=np.array([271.5, 232, 509.8]).reshape(3,1)
    w, b=weights_init(10,20,30,40,50)
    w, b=train(x_norm,y_train,w,b,0.1,10000,0.01)
    x_in=get_data(mean,std)
    y_pred=(x_in@w.T+b).reshape(1)[0]
    print(f"Predicted value: {y_pred}")
    
def gd_scikit():
    path='D:\Engineer\Machine Learning Andrew NG\Machine-Learning-Specialization-Coursera\C1 - Supervised Machine Learning - Regression and Classification\week2\Optional Labs\data\houses.txt'
    x_train_0, x_train_1, x_train_2, x_train_3, y_train=load_data(path)
    x_train, x_norm, mean, std=data_init(x_train_0,x_train_1,x_train_2,x_train_3)
    scaler=StandardScaler()
    X_norm=scaler.fit_transform(x_train)
    sgdr=SGDRegressor(max_iter=10000, tol=0.0001)
    sgdr.fit(X_norm,y_train)
    w_norm=sgdr.coef_
    b_norm=sgdr.intercept_
    print(f"sgdr: {sgdr}, iterations: {sgdr.n_iter_}, w: {w_norm}, b: {b_norm}")
    # x_in=get_data(mean,std)
    # y_pred=sgdr.predict(x_in)
    # print(f"Predicted value: {y_pred}")
    y_pred=sgdr.predict(X_norm)
    print(f"Predicted value: {y_pred[1:5]}\nActual value:    {y_train[1:5].reshape(4,)}")
    
def linear_regression_scikit():
    # x_train=np.array([1, 2]).reshape((-1,1))
    # y_train=np.array([300, 500]).reshape((-1,1))
    path='D:\Engineer\Machine Learning Andrew NG\Machine-Learning-Specialization-Coursera\C1 - Supervised Machine Learning - Regression and Classification\week2\Optional Labs\data\houses.txt'
    x_train_0, x_train_1, x_train_2, x_train_3, y_train=load_data(path)
    x_train, x_norm, mean, std=data_init(x_train_0,x_train_1,x_train_2,x_train_3)
    linear_model=LinearRegression()
    linear_model.fit(x_train,y_train)
    w=linear_model.coef_
    b=linear_model.intercept_
    y_pred=linear_model.predict(x_train)
    print(f"Predicted value: {y_pred[8:12].reshape((-1,))}\nActual value: {y_train[8:12].reshape((-1,))}\nW: {w}, B: {b}")
    
def linear_regression_hossam_hassan():
    path='D:\Engineer\Machine Learning Andrew NG\Machine-Learning-Specialization-Coursera\C1 - Supervised Machine Learning - Regression and Classification\week2\Optional Labs\data\houses.txt'
    x_train_0, x_train_1, x_train_2, x_train_3, y_train=load_data(path)
    x_train=np.array([x_train_0, x_train_1, x_train_2, x_train_3]).T
    mean=np.mean(x_train,axis=0)
    std=np.std(x_train,axis=0)
    x_norm=(x_train-mean)/std
    x_norm=np.hstack((x_norm,np.ones((len(x_norm),1)))) #[x0,x1,x2,x3,1]
    w=np.array([1,2,3,4,5]).reshape(5,1) #[w0,w1,w2,w3,b]
    w=linear_regression_hossam_hassan_gd(x_norm, y_train, w, 0.01, 10000, 0.01)
    y_pred=x_norm@w
    # print((y_train==y_pred).all())
    print(f"Predicted value: {y_pred[1:5].reshape((-1,))}\nActual value: {y_train[1:5].reshape((-1,))}\nW: {w}")
    
def linear_regression_hossam_hassan_gd(x, y, w, r=0.01, epoch=100, tol=0.01):
    m=len(y)
    Epoch=epoch
    while epoch>=0:
        y_pred=x@w
        loss=(1/(2*m))*np.sum((y_pred-y)**2)
        print(f"Epoch: {Epoch-epoch}, Loss: {loss}, w: {w.T}")
        if loss<tol:
            break
        dw=(1/m)*(x.T@(y_pred-y))
        w=w-r*dw
        epoch-=1
    return w
    
if __name__=="__main__":
    #gd_handmade()
    #gd_scikit()
    #linear_regression_scikit()
    linear_regression_hossam_hassan()