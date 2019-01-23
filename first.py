import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_=pd.read_csv('seismic.csv',header=None)
data=data_.drop_duplicates(subset=None, keep='first', inplace=False)


#data=data.drop(labels=[0,1,2,5,6,7,8,9],axis=1)




data.head()



fig , ax = plt.subplots(figsize=(4,4))
sns.countplot(x=18, data=data,color='blue')
plt.title("Count of hazardous and Non Hazardous")
plt.show()



X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values




from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

X[:,1] = labelencoder.fit_transform(X[:, 1])

X[:,7] = labelencoder.fit_transform(X[:, 7])



X[:,2] = labelencoder.fit_transform(X[:, 2])

X=np.array(X,dtype=float)


num=[0,1,2,3,4,5,6]
corr_df=data[num] 
cor= corr_df.corr(method='pearson')
print(cor)



ig, ax =plt.subplots(figsize=(8, 6))
plt.title("Correlation Plot")
sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 1)
classifier.fit(X_train, y_train)



cool=classifier.decision_path(X_train)


cool1=cool[0].todense()

cool2=classifier.decision_path(X_test)
cool3=cool2[0].todense()




import keras
from keras.models import Sequential
from keras.layers import Dense


classifier2 = Sequential()

classifier2.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu', input_dim = 3382))

classifier2.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu'))

classifier2.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
   
classifier2.fit(cool1, y_train, batch_size = 10, nb_epoch = 5)


y_pred = classifier2.predict(cool3)
y_pred = (y_pred > 0.5)
y_pred=labelencoder.fit_transform(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

count=0
for i in range(516):
    if y_pred[i]==y_test[i]:
        count+=1
print('accuracy:'+str((count/len(y_test)*100))+'%')

test_data=np.array(X_test[109]).reshape((1,-1))




#testing a real time data
cool_test=classifier.decision_path(test_data)
test_cool=cool_test[0].todense()
real_pred=classifier2.predict(test_cool)
real_pred = (real_pred > 0.5)
real_pred=labelencoder.fit_transform(real_pred)



loss=[0.2289,0.1327,0.0605,0.0295,0.0094]
accu=[0.9321,0.9520,0.9782,0.9893,0.9971]
num_iters=np.arange(5)
plt.plot(num_iters,loss,color='red')
plt.xlabel('number of Iterations')
plt.ylabel('loss')
plt.title('variation of loss wrt no of iterations')
plt.show()


plt.plot(num_iters,accu,color='green')
plt.xlabel('number of iterations')
plt.ylabel('accuracy')
plt.title('Variation of accuracy wrt no of iterations')


precision= (475/(475+7))*100
recall=(475/(475+32))*100


eval_results = 2 * ( (0.9971* 0.93688) / ( 0.9854 + 0.93688 ))





