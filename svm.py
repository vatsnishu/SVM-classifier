#!/usr/bin/python

from datetime import date
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.svm import SVC

Cs=[5,50,100]
degrees=[2,3,4]

dataset=pandas.read_csv('Data_SVM.csv')
classes=dataset.values
maxFinalAccuracyMean=0.0
maxFinalStDev=0.0
cmax=0
degreesmax=0
print "C           degree          Accuracy Mean            Accuracy_Standard_Deviation"
print"_________________________________________________________________________________"

#Training Non-Linear SVM classifier using rbf(Gaussian Kernel)
for Ci in Cs:
    for degreei in degrees:

        n_epochs=30
        finalAccuracyMean=0.0
        finalAccuracyStdev=0.0
        for i in range(n_epochs):
            features=dataset.iloc[:,: -1].values
            labels=dataset.iloc[:, 2].values

            x_tr,x_test,y_tr,y_test=train_test_split(features,labels,test_size=0.2,random_state=0)

            

            svmPoly=SVC(C=Ci,kernel='rbf',random_state=0,degree=degreei)
            svmPoly.fit(x_tr,y_tr)
    
            y_pred=svmPoly.predict(x_test)

            accuracies=cross_val_score(estimator=svmPoly,X=x_tr,y=y_tr,cv=10)
            finalAccuracyMean+=accuracies.mean()
            finalAccuracyStdev+=accuracies.std()

        finalAccuracyMean/=30
        finalAccuracyStdev/=30
        print str(Ci)+"           "+str(degreei)+"              "+str(finalAccuracyMean)+"            "+str(finalAccuracyStdev)

        if(finalAccuracyMean>maxFinalAccuracyMean):
            cmax=Ci
            degreesmax=degreei
            maxFinalAccuracyMean=finalAccuracyMean
            maxFinalStDev=finalAccuracyStdev

#Running SVM using rbf and C and D found corresponding to maximum accuracy
svmPoly=SVC(C=cmax,kernel='rbf',random_state=0,degree=degreesmax)

features=dataset.iloc[:,: -1].values
labels=dataset.iloc[:, 2].values

svmPoly.fit(features,labels)

accuracies=cross_val_score(estimator=svmPoly,X=features,y=labels,cv=10)

print
print "Maximum Accuracy mean corresponding to C ="+str(cmax)+"and d= "+str(degreesmax)+"is:"+str(maxFinalAccuracyMean)
print "Maximum Accuracy mean corresponding to C ="+str(cmax)+"and d= "+str(degreesmax)+"is:"+str(maxFinalStDev)


#Plotting the results
X_set,Y_set=features,labels
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,svmPoly.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(Y_set)):

    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)

plt.title('SVM Classification {Test Set}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()




