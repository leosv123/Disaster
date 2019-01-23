import numpy as np
import pandas as pd


data_=pd.read_csv('seismic.csv',header=None)
data=data_.drop_duplicates(subset=None, keep='first', inplace=False)


#data=data.drop(labels=[0,1,2,5,6,7,8,9],axis=1)




X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values




from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

X[:,1] = labelencoder.fit_transform(X[:, 1])

X[:,7] = labelencoder.fit_transform(X[:, 7])



X[:,2] = labelencoder.fit_transform(X[:, 2])

X=np.array(X,dtype=float)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(X_train,y_train)


pred = lassoReg.predict(X_test)

lassoReg.score(X_test,y_test)









from sklearn.decomposition import PCA

# Make an instance of the Model
pca = PCA(.8)

pca.fit(X_train)
X_train= pca.transform(X_train)
X_test = pca.transform(X_test)
"""

#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#classifier.fit(X_train, y_train)
#
#
## Predicting the Test set results
#y_pred = classifier.predict(X_test)
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred)
#count=0
#for i in range(516):
#    if y_pred[i]==y_test[i]:
#        count+=1
#print(str((count/len(y_test)*100))+'%')







#cool=classifier.predict(np.array(X_test[109]).reshape((1,-1)))



#
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()


#
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#



# Fitting Random Forest Classification to the Training set,random state must be 1 for RandomforestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 1)
classifier.fit(X_train, y_train)



cool=classifier.decision_path(X_train)


cool1=cool[0].todense()

cool2=classifier.decision_path(X_test)
cool3=cool2[0].todense()

# Predicting the Test set results
#y_pred = classifier.predict(cool3)

## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)



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



