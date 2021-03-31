import numpy as np
import matplotlib.pyplot as plt
import time
import copy

plot=True # flag to compare runtime

data = np.loadtxt('data.txt') #loading data
X = np.array(data[:, 0:8]) # features are first 8 columns
y = np.array(data[:, 8], ndmin=2).T # last column is the label


#given parameters:

lr_bgd = 0.000000001
ep_bgd = 0.03

lr_sgd = 0.00000001#0.00000001
ep_sgd = 0.4

lr_mbgd = 0.00000001
ep_mbgd = 0.5



def loss(X,weights,y,b,C): # calculates cost
    
    # X=mxn, weights=nx1
    J=C*np.sum(np.maximum(0,(1-y.T*(np.dot(X,weights)+b).T)))+ 0.5*np.dot(weights.T,weights)
    
    return J.item()

def J_w(X,y,b,weights,C): # calculates gradient w.r.t wj
    
    #dw=copy.deepcopy(weights)
    n=X.shape[1]
    dw = np.zeros((n,1))
    if X.shape[0]>1: # X is mxn matrix
        for j in range (X.shape[1]):
            dw[j]-=C*np.sum((y.T*X[:,j])*(y.T*((np.dot(X,weights)+b).T)<1))
    elif X.shape[0]==1: # X is a row vector
        for j in range (X.shape[1]):
            dw[j]-=C*((y*X[0,j])*(y*((np.dot(X[0,:],weights)+b).T)<1))
    
    return dw

def J_b(X,y,b,weights,C): # calculates gradient w.r.t b
    
    if X.shape[0]>1:
        db=-C*np.sum(y.T*(y.T*((np.dot(X,weights)+b).T)<1))
    elif X.shape[0]==1:
        db=-C*(y*(y*((np.dot(X[0,:],weights)+b).T)<1))
         
    return db

def bgd(X,y,lr,ep): # function for batch gradient descent
    
    n = X.shape[1] # no. of features
    m = X.shape[0] # no. of examples
    C = 10 # given in problem
    b = 0 # bias initialization
    weights = np.zeros((n,1)) # weights initialization
    costHistory = [] # to store cost for plotting
    
    
    costHistory.append(loss(X, weights, y, b,C))

    k = 0
    while True:
        grad = J_w(X,y,b,weights,C) #calculating gradient for weights
        grad_b = J_b(X,y,b,weights,C) #calculating gradient for bias
        
        b = b - lr*grad_b #update b
        weights = weights - lr*grad #update weights
        
        cost=loss(X, weights, y, b,C)
        costHistory.append(cost)
        k = k+1
        
        # convergence critrion
        if abs(costHistory[k-1]-costHistory[k])*100/costHistory[k-1] < ep:
            break
    return costHistory, weights


bgd_start_time= time.time()
cost_bgd, weights_bgd = bgd(X, y, lr_bgd, ep_bgd)
bgd_end_time=time.time()
print('BGD convergence time: ',bgd_end_time-bgd_start_time)
print('Converged with cost: ',cost_bgd[-1])
#plt.plot(cost_bgd)
#plt.ylabel('BGD cost')
#plt.show()

def sgd(X,y,lr,ep):
    
    n = X.shape[1] # no. of features
    m = X.shape[0] # no. of examples
    C = 10 # given in problem
    b = 0 # bias initialization
    weights = np.zeros((n,1)) # weights initialization 
    costHistory = [] # to store cost for plotting
    
    costHistory.append(loss(X, weights, y, b,C))
    k = 0
    l=0
    
    while True:
        for i in range (m):
            x=X[i:i+1,:]
            Y=y[i]
            grad = J_w(x,Y,b,weights,C) #calculating gradient for weights
            grad_b = J_b(x,Y,b,weights,C) #calculating gradient for bias
        
            b = b - lr*grad_b #update b
            weights = weights - lr*grad #update weights
            if plot==True:
                cost=loss(X, weights, y, b,C)
                costHistory.append(cost)
                k = k+1
        if plot==False:
            k+=1
            cost=loss(X, weights, y, b,C)
            costHistory.append(loss(X, weights, y, b,C))
            if abs(costHistory[k-1]-costHistory[k])*100/costHistory[k-1] < ep:
                break
            
        else:
            if abs(costHistory[l]-costHistory[k-1])*100/costHistory[l] < ep:
                break
            l=k-1
    return costHistory, weights


sgd_start_time= time.time()
cost_sgd, weights_sgd = sgd(X, y, lr_sgd, ep_sgd)
sgd_end_time=time.time()
print('SGD convergence time: ',sgd_end_time-sgd_start_time)
print('Converged with cost: ',cost_sgd[-1])
#plt.plot(cost_sgd)
#plt.ylabel('SGD cost')
#plt.show()


def mbgd(X,y,lr,ep,batch_size):
    
    n = X.shape[1] # no. of features
    m = X.shape[0] # no. of examples
    C = 10 # given in problem
    b = 0 # bias initialization
    weights = np.zeros((n,1)) # weights initialization 
    costHistory = [] # to store cost for plotting
    
    costHistory.append(loss(X, weights, y, b,C))
    
    k = 0
    l=0
    while True:
        
        for i in range (0,m-batch_size+1, batch_size):
            x = X[i:i+batch_size,:]
            Y = y[i:i+batch_size]
            
            grad = J_w(x,Y,b,weights,C) #calculating gradient for weights
            grad_b = J_b(x,Y,b,weights,C) #calculating gradient for bias
        
            b = b - lr*grad_b #update b
            weights = weights - lr*grad #update weights
            if plot==True:
                cost=loss(X, weights, y, b,C)
                costHistory.append(cost)
                k = k+1
                
        if plot==False:
            k+=1
            cost=loss(X, weights, y, b,C)
            costHistory.append(loss(X, weights, y, b,C))
            if abs(costHistory[k-1]-costHistory[k])*100/costHistory[k-1] < ep:
                break
            
            
        else:
            if abs(costHistory[l]-costHistory[k-1])*100/costHistory[l] < ep:
                break
            l=k-1
            
    return costHistory, weights


mbgd_start_time= time.time()
cost_mbgd, weights_mbgd = mbgd(X, y, lr_mbgd, ep_mbgd,batch_size=4)
mbgd_end_time=time.time()
print('MBGD convergence time: ',mbgd_end_time-mbgd_start_time)
print('Converged with cost: ',cost_mbgd[-1])
#plt.plot(cost_mbgd)
#plt.ylabel('MBGD cost')
#plt.show()


# combined plot for visualization
if plot==True:
    fig, ax = plt.subplots()
    iterations_bgd=[i+1 for i in range(len(cost_bgd))]
    ax.plot(iterations_bgd, cost_bgd, 'k--', label='Batch GD')
    iterations_sgd=[i+1 for i in range(len(cost_sgd))]
    ax.plot(iterations_sgd, cost_sgd, label='Stochastic GD')
    iterations_mbgd=[i+1 for i in range(len(cost_mbgd))]
    ax.plot(iterations_mbgd, cost_mbgd, label='Mini Batch GD')
    iters=[i for i in range(0,4500,500)]
    ax.set_xticks(iters)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show()
