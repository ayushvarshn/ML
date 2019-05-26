import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('lr.csv')
x1=list(df['A'])
x2=list(df['B'])
y=list(df['C'])
n=float(len(y))
x1_pos=[x1[i] for i in range(len(y)) if(y[i]==1)]
x1_neg=[x1[i] for i in range(len(y)) if(y[i]==0)]
x2_pos=[x2[i] for i in range(len(y)) if(y[i]==1)]
x2_neg=[x2[i] for i in range(len(y)) if(y[i]==0)]
plt.scatter(x1_pos,x2_pos,color='red')    
plt.scatter(x1_neg,x2_neg)
  

w1 = 0
w2=0
b = 0
def model(x1,x2,w1,w2,b):
    h=1.0/(1+np.exp(-b-(w1*x1)-(w2*x2)))
    return h
  
def grad_update(x1,x2,y,w1,w2,b,learning_rate = 0.0005):
  n=float(len(y))
  sum_dw1=0
  sum_dw2=0
  sum_db=0
  for i in range(len(y)):
    h=model(x1[i],x2[i],w1,w2,b)
    db=-(y[i]-h)
    dw1=-(y[i]-h)*x1[i]
    dw2=-(y[i]-h)*x2[i]
    sum_dw1+=dw1
    sum_dw2+=dw2
    sum_db+=db
  sum_dw1=(2/n)*(sum_dw1)
  sum_dw2=(2/n)*(sum_dw2)
  sum_db=(2/n)*(sum_db)
  w2=w2-learning_rate*(sum_dw2)
  w1=w1-learning_rate*(sum_dw1)
  b=b-learning_rate*(sum_db)
  h=model(x1[i],x2[i],w1,w2,b)
  cost=-(1/n)*(y[i]*float(np.log(h))+(1-y[i])*float(np.log(1-h)))
  print(cost)
  return w1,w2,b                                              

lis=[]
for i in range(100000):   
  w1,w2,b = grad_update(x1,x2,y,w1,w2,b) 
for i in range(len(y)):
  lis.append(-(b+(w1*x1[i]))/w2)

print(w1,w2,b)

plt.plot(x1,lis)
plt.show()
