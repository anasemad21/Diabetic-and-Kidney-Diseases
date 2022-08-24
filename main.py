# TEAM 20188009  20188022  20188036
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('diabetic_kidney_disease.csv')
data=(data-data.min())/(data.max()-data.min())
data.insert(0,'ones',1)# insert column of ones to handle matrix mult and the difference between theta 0 and thata 1
#spiliting data to x and y
columns=data.shape[1]
X=data.iloc[:91,0:columns-3] # to split data this is X values
Y=data.iloc[ :91,columns-1:columns] # this is Y values

X=data.iloc[:91,0:columns-3] # to split data this is X values
Y=data.iloc[ :91,columns-1:columns] # this is Y values
X=np.matrix(X.values) # to convert to matrix
Y=np.matrix(Y.values)  # to convert to matrix
theta=np.matrix(np.array([0,0]))  # to convert to matrix

X_test=data.iloc[91:,0:columns-3] # to split data this is X values
Y_test=data.iloc[ 91:,columns-1:columns] # this is Y values
X_test=np.matrix(X_test.values) # to convert to matrix
Y_test=np.matrix(Y_test.values)  # to convert to matrix

def cost(X,Y,theta) :
    c=np.power((X * theta.T)-Y ,2)
    return np.sum(c)/(2 *len(X))

def GredientDec(X,Y,theta,alpha,num_iter):
    t=np.matrix(np.zeros(theta.shape))
    num_theta=int(theta.ravel().shape[1])
    costs=np.zeros(num_iter)
    i=0
    j=0
    while i != num_iter :
        errors=(( X*theta.T)-Y)
        j=0
        while j != num_theta :
            trm=np.multiply(errors,X[: ,j])
            t[0,j]=t[0,j]-((alpha/(len(X)) * np.sum(trm)))
            j+=1
        theta=t
        costs[i]=cost(X,Y,theta)
        i+=1
    return theta,costs
zb=theta[0,0]+theta[0,1]*X #intial value od theta 0 and theta 1 for drawing line before gradient decent

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(X,zb,'r',label="prediction Before")
ax.scatter(data['FBG (mg/dL)'],data['UACR (mg/g creatinine)'],label="trainig Data")
ax.legend(loc=2)
ax.set_xlabel('FBG')
ax.set_ylabel('UACR')
ax.set_title('Line Before Optimization ')
plt.show()

alpha=0.1
num_iter=1000
f,costt=GredientDec(X,Y,theta,alpha,num_iter)
#print("f \n",f)
z=f[0,0]+(f[0,1]*X)
#print('cost \n',costt[:])

def Compute_y_predict(X_test,theta):
    return (X_test*theta.T)
print(Compute_y_predict(X_test,f))

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(X,z,'g',label="preddection")
ax.scatter(data['FBG (mg/dL)'],data['UACR (mg/g creatinine)'],label="trainig Data")
ax.legend(loc=2)
ax.set_xlabel('FBG')
ax.set_ylabel('UACR')
ax.set_title('Line After Optimization')
plt.show()




