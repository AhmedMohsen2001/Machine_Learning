import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')

def print_data(x, y):
    for i in range(len(x)):
        print(f"x^({i}): {x[i]}, y^({i}): {y[i]}")

def scatter_data(x, y, color='r'):
    plt.scatter(x, y, marker='x', c=color)
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Scatter Plot of Data')
    
def plot_data(x, y, color='r'):
    plt.plot(x, y, marker='x', c=color)
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('Plot of Data')
    
def train(x, y, w, b, r=0.01, epoch=100, tol=0.01):
    m=len(x)
    Epoch=epoch
    while epoch>=0:
        loss=1/(2*m)*np.sum((w*x+b-y)**2)
        print(f"Epoch: {Epoch-epoch}, Loss: {loss}, w: {w}, b: {b}")
        if loss<tol:
            break
        w_new=w-r*(1/m)*np.sum(x*(w*x+b-y))
        b_new=b-r*(1/m)*np.sum(w*x+b-y)
        w=w_new
        b=b_new
        epoch-=1
        if epoch%100==0:
            plt.plot(Epoch-epoch-1,loss,marker='x',c='b')
    plt.show()
    return w, b

if __name__=="__main__":
    x_train=np.array([1.0, 2.0])
    y_train=np.array([300, 500])
    # x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    # y_train = np.array([250, 300, 480,  430,   630, 730,])
    w=100
    b=100
    m=len(x_train)
    #print(f"x_train: {x_train}\ny_train: {y_train}\nSize of training data: {m}")
    print_data(x_train,y_train)
    scatter_data(x_train,y_train,'r')
    y_expected=w*x_train+b
    plot_data(x_train,y_expected,'b')
    w, b=train(x_train,y_train,w,b,0.01,10000,0.001)
    y_expected=w*x_train+b
    plot_data(x_train,y_expected,'g')
    
    plt.show()