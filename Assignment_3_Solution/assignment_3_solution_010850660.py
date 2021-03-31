import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

debug=False # flag to debug

lamb_=3 # for regularization term in loss function (given in question)
lr=0.2 #learning rate (given in question)
iter_num=500 #given number of iterations for batch gradient
input_features = pd.read_csv('./data/X.csv',index_col=False,header= None)
X = input_features.to_numpy() # input_matrix 5000X400
output_labels = pd.read_csv('./data/Y.csv',index_col = False,header = None)
Y = output_labels.to_numpy() # given labels vector 5000X1

# function for one hot encoding
def one_hot_vector_encoder(Y):
    row_num = Y.shape[0]
    col_num = 10 # 10 digits
    one_hot_vector = np.zeros([row_num,col_num],dtype=int)
    for i in range(one_hot_vector.shape[0]):
        for j in range(one_hot_vector.shape[1]+1):
            one_index=Y[i,0]-1 # Since 1 should be labeled at 0th index and 0 should be labeled as 10th index
            if j==one_index:
                one_hot_vector[i,j]=1
    return one_hot_vector

Y= one_hot_vector_encoder(Y) #one_hot_encoded (shape=5000X10)
b = np.ones([X.shape[0],1]) # bias initialization (5000X1) value =1

## Loading given Weight matrix
init_w1 = pd.read_csv('./data/initial_W1.csv',index_col = False, header = None)
init_w2 = pd.read_csv('./data/initial_W2.csv',index_col = False, header = None)
initial_W1 = init_w1.to_numpy() # weight matrix for hidden layer (25X401)
initial_W2 = init_w2.to_numpy() # weight matrix for hidden layer (10X26)


# Accuracy checking
def calc_accuracy(Y_actual, Y_pred):
    pred = np.argmax(Y_pred,axis = 1)
    actual = np.argmax(Y_actual,axis =1)
    loss = sum([x != y for x,y in zip (pred,actual)])
    count = len(pred)
    accuracy= 1- (loss/count)
    return accuracy*100

#activation function
def logistic(z):
    
    return  1/(1 + np.exp(-z))

# Gradient of logistic function
def grad_logistic(z):
    
    return (logistic(z)*(1 - logistic(z)))


#Forward propagation
def forward_propagation(X,b,W1,W2):
    X = np.concatenate((b,X),axis=1) # adding bias column vector to make X shape (5000X401) compatible with W1
    Z1 = np.dot(X,W1.T) # X=5000X401, W1=25X401. So Z1=5000X25
    H = logistic(Z1)
    H = np.concatenate((b,H), axis = 1) # adding bias column vector to make H shape (5000X26) compatible with W2
    Z2 = np.dot(H,W2.T) # H=5000X26, W2=10X26. So Z2=5000X10
    Y_pred = logistic(Z2)
    
    return Y_pred, H, Z1 #3 returnig H and Z1 to be used in backward propagation

if debug:
    ## Debugging Forward Propagation part
    W1 = pd.read_csv('./data/W1.csv', index_col= False, header = None)
    forward_W1 = W1.to_numpy()
    W2 = pd.read_csv('./data/W2.csv', index_col = False, header = None)   
    forward_W2 = W2.to_numpy() 
    forward_Y,forward_H,forward_Z1 = forward_propagation(X,b,forward_W1,forward_W2)

    accuracy = calc_accuracy(Y, forward_Y)
    print("Accuracy in Forward Propagation Debugging: {}% ".format(accuracy))

    ##----------------------------------------------------------------------------

##loss function
def loss_function(Y_actual,Y_pred,lamb_,W1,W2):
    m= Y_pred.shape[0] # 5000: no. of examples
    ones_vector= np.ones([len(Y_actual),1],dtype=int)
    first_term = 1/m*((np.sum(-Y_actual * np.log(Y_pred) - (ones_vector - Y_actual)* np.log(ones_vector - Y_pred))))
    W1_wo_b=W1[:,1:W1.shape[1]]
    W2_wo_b=W2[:,1:W2.shape[1]]
    second_term=lamb_/(2 * m)*(np.sum((W1_wo_b)**2) + np.sum((W2_wo_b)**2))
    return first_term + second_term 

if debug:
    ## ----------debugging loss function implementation------------
    cost=loss_function(Y,forward_Y,lamb_,forward_W1,forward_W2)
    print("Cost after forward propagation: ",cost)
    ##----------------------------------------------------------------------------------------------------------
    

# Back propagation
def back_propagation(Y_pred, Y_actual,X,b,W1,W2,H,Z1,lamb_):
    X = np.concatenate((b,X),axis=1)
    regu_W1 = W1
    regu_W1[:,0] = 0 #first column =0
    regu_W2 = W2
    regu_W2[:,0] = 0 # first column =0
    beta2 = Y_pred - Y_actual
    beta1 = np.dot(beta2,W2[:,1:])*grad_logistic(Z1)
    dW2 = 1/len(Y_pred)*(np.dot(beta2.T,H) + lamb_*(regu_W2))
    dW1 = 1/len(Y_pred)*(np.dot(beta1.T,X) + lamb_*(regu_W1))
    return dW1, dW2

def batch_grad(X,b,Y,W1,W2,lamb_, lr, iter_num,debug):
    costHistory = []
    Y_pred_1,h,z = forward_propagation(X,b,W1,W2)
    cost = loss_function(Y, Y_pred_1,lamb_,W1,W2)
    costHistory.append(cost)
    for i in range(iter_num):
        Y_pred, H, Z1 = forward_propagation(X,b,W1,W2)
        dW1, dW2 = back_propagation(Y_pred,Y,X,b,W1,W2,H,Z1,lamb_)
        if debug==True:
            np.savetxt("./data/W1_result_"+str(i)+".csv", dW1, delimiter=",")
            np.savetxt("./data/W2_result_"+str(i)+".csv", dW2, delimiter=",")
        W1 = W1 - lr*dW1
        W2 = W2 - lr*dW2
        cost = loss_function(Y,Y_pred,lamb_,W1,W2)
        costHistory.append(cost)
        
    return costHistory,W1,W2
## ------------------------------------------------Back propagation debugging -------------------------
if debug:
    cost,W1,W2=batch_grad(X,b,Y,initial_W1,initial_W2,lamb_, lr,3,True)

# Calling batch_gradient to get results after 500 iterations
final_cost,W1,W2=batch_grad(X,b,Y,initial_W1,initial_W2,lamb_, lr,iter_num,False)
plt.plot(final_cost)
plt.ylabel('Loss Function')
plt.xlabel('Number of Iterations')
plt.title("Loss vs number of iterations")
plt.show()

#checking accuracy after training the weights 
Pred_Y,H,Z1 = forward_propagation(X,b,W1,W2)
accuracy = calc_accuracy(Y, Pred_Y)
print("Accuracy after 500 iterations: {}% ".format(accuracy))

# checking the 10 examples given in the question
print("Checking the given indices:  ")
check_index=[2171,145 ,1582, 2446 ,3393, 815, 1378, 529, 3945, 4628]
for i in range(len(check_index)):
    print("index:",check_index[i], "Prediction:", np.where(Pred_Y[check_index[i]-1]==max(Pred_Y[check_index[i]-1]))[0]+1)
