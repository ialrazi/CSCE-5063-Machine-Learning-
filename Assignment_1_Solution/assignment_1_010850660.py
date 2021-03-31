import numpy as np
import matplotlib.pyplot as plt



def squared_loss(X, y, theta):
    part = y-np.dot(X, theta)
    cost = (1/m1)*np.dot(part.T, part)
    return cost.item()
def loss_quad(X, y, theta, la_q):
    part = y-np.dot(X, theta)
    cost = 1/(2*m)*np.dot(part.T, part) + la_q/(2*m)*np.dot(theta.T, theta)
    return cost.item()
def loss_lasso(X, y, theta, la_l):
    part = y-np.dot(X, theta)
    cols_one=np.ones((X.shape[1],1))
    cost = 1/(2*m)*np.dot(part.T, part) + la_l/(2*m)*np.dot((np.absolute(theta)).T,cols_one)
    #print("C",cost)
    return cost.item()

def gradientDescent(X, y, alpha, ep, la):
    m = X.shape[0]
    n = X.shape[1]
    para = np.zeros((n,1))
    new_para = np.zeros((n,1))
    costHistory = []
    if la==5: # quadratic
        costHistory.append(loss_quad(X, y, para, la))
    if la==1: # lasso
        costHistory.append(loss_lasso(X, y, para, la))
    k = 0
    while True:
        h = (np.dot(X, para)-y).T # compute the predictions of all training examples.
        for j in range(n):
            if la==5: # quadartic
                grad = 1/m*np.dot(h, X[:, j]) + la/m*para[j]
            if la==1: # lasso
                if para[j]<0:
                    grad = 1/m*np.dot(h, X[:, j]) - la/(2*m)
                else:
                    grad = 1/m*np.dot(h, X[:, j]) + la/(2*m)
                
            new_para[j] = para[j] - alpha*grad # store the new paratemeters in a new vector.
        para = new_para
        
        if la==5: # quadratic
            costHistory.append(loss_quad(X, y, para, la))
        if la==1: #lasso
            costHistory.append(loss_lasso(X, y, para, la))
        k = k+1
        
        # convergence critrion
        
        if abs(costHistory[k-1]-costHistory[k])*100/costHistory[k-1] < ep:
            break
		
    return costHistory, para





data = np.loadtxt('data.txt')
X = np.array(data[:, 1:16]) # feature matrix
y = np.array(data[:, 16], ndmin=2).T # target
len_feature=X.shape[1] # number of features for each data
len_data= X.shape[0] # number of data

for i in range(len_feature):
	X[:,i]=(X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i])) # normalization for feature data
y=(y-np.min(y))/(np.max(y)-np.min(y)) # normalizing target data

train_len=int(0.8*len_data) # 80% training data
train_X=X[0:train_len,:] # training data set for features
test_X=X[train_len:,:] # testing data set for features
train_y=y[0:train_len] # training target data set
test_y=y[train_len:] # testing target data
m=train_len
m1=len(test_X)


# adding 1 (coeffficient corresponds to theta_0) for each example
cols_one=np.ones((train_len,1))
train_X=np.concatenate((cols_one,train_X),axis=1) # now the feature numbers has increased by 1 for each example
theta=np.zeros((train_X.shape[1],1)) # vector of parameters (theta) initialized with 0
alpha=0.01 # learning rate
ep=0.001  #error tolerance
la_q=5 # regularization coefficient for quadratic
la_l=1 # regularization coefficient for lasso

cost_quad, para_quad = gradientDescent(train_X, train_y, alpha, ep, la_q)
cost_lasso, para_lasso = gradientDescent(train_X, train_y, alpha, ep, la_l)
cols_one=np.ones((test_X.shape[0],1))


test_X=np.concatenate((cols_one,test_X),axis=1) 

squared_loss_w_quad=squared_loss(test_X,test_y,para_quad)
squared_loss_w_lasso=squared_loss(test_X,test_y,para_lasso)

print("LASSO",squared_loss_w_lasso)
print("QUAD",squared_loss_w_quad)

non_zero_count_quad=len(para_quad)
non_zero_count_lasso=len(para_lasso)
total_quad_params=non_zero_count_quad
total_lasso_params=non_zero_count_lasso
print("Total_Lasso_Param",total_lasso_params)
print("Total_Quad_Param",total_quad_params)

for i in range(len(para_quad)):
	if abs(para_quad[i])<0.01:
		non_zero_count_quad-=1
for i in range(len(para_lasso)):
	if abs(para_lasso[i])<0.01:
		non_zero_count_lasso-=1  

print("Zero_Lasso_Param",total_lasso_params-non_zero_count_lasso)
print("Zero_Quad_Param",total_quad_params-non_zero_count_quad)

#individual plot according to instructions
plt.plot(cost_quad)
plt.ylabel('Cost for Quadratic Regularization')
plt.xlabel('Iteration')
plt.show()

plt.plot(cost_lasso)
plt.ylabel('Cost for Lasso Regularization')
plt.xlabel('Iteration')
plt.show()
# combined plot for visualization
fig, ax = plt.subplots()
iterations_quad=[i+1 for i in range(len(cost_quad))]
ax.plot(iterations_quad, cost_quad, 'k--', label='Quadratic Regularization')
iterations_lasso=[i+1 for i in range(len(cost_lasso))]
ax.plot(iterations_lasso, cost_lasso, label='Lasso Regularization')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()
