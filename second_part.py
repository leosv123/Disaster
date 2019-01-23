import pandas as pd

import numpy as np

import tensorflow as tf
import cv2

path2='C:\\Users\\lingraj\\Downloads\\ABCDdataset\\resized\\patch-pairs\\'
path='C:\\Users\\lingraj\\Downloads\\ABCDdataset\\resized\\5fold-list\\'
train_data = pd.read_csv(tf.gfile.Open(path+'cv1-train.csv'),header=None)
test_data= pd.read_csv(tf.gfile.Open(path+'cv1-test.csv'),header=None)
tr_data=list()
te_data=list()
for i in range(6752):
    img=cv2.imread(path2+train_data.iloc[i,0]).reshape(1,49152)
    tr_data.append(img)

data_train=np.array(tr_data).reshape(6752,49152)

for i in range(1692):
    img=cv2.imread(path2+test_data.iloc[i,0]).reshape(1,49152)
    te_data.append(img)

data_test=np.array(te_data).reshape(1692,49152)




X_train=pd.DataFrame(data_train)
X_test=pd.DataFrame(data_test)

X_train.to_csv('X_train.csv')


X_train=X_train.values.reshape(X_train.shape[0],128,128,3)#.astype(np.float32)
X_test=X_test.values.reshape(X_test.shape[0],128,128,3)#.astype(np.float32)
y_train=np.array(train_data.iloc[:,1])
y_test=np.array(test_data.iloc[:,1])
y_train=np.array(train_data.iloc[:,1]).reshape(len(y_train),1)
y_test=np.array(test_data.iloc[:,1]).reshape(len(y_test),1)
y_train=np.array(train_data.iloc[:,1]).reshape(len(y_train),1)
y_test=np.array(test_data.iloc[:,1]).reshape(len(y_test),1)




