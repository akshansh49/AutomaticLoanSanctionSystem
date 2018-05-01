
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
# Importing the dataset

def preprocess_():
    
        f1 = open('train_.csv', "r+")
        f2 = open('train_predone1.csv','w+')
        for i in f1 :
        	j = i.strip().split(',')
        	
        	if j[1] == "Male" :
        		j[1] = '1'
        	else :
        		j[1] = '0'
        
        	if j[2] == 'Yes' :
        		j[2] = '1'
        	else :
        		j[2] = '0'
        
        	j[3] = j[3].replace('+','')#.replace('','0')
        
        	if j[4] == 'Graduate' :
        		j[4] = '1'
        	else :
        		j[4] = '0'	 
        
        	if j[5] == 'Yes' :
        		j[5] = '1'
        	else :
        		j[5] = '0'
        	
        	if j[11] == 'Urban' :	
        		j[11] = '1'	
        	elif j[11] == 'Semiurban' :	
        		j[11] = '2'
        	else :
        		j[11] = '0'
        
        	if j[12] == 'Y' :
        		j[12] = '1'
        	else :
        		j[12] = '0'
                
        
        	k = ','.join(j)
        	f2.write(k+'\n')
    
#preprocess_()

df = pd.read_csv('/Users/mohitbindal/Desktop/topl project/train_predone.csv')

df.describe()
df.LoanAmount.fillna(np.mean(df.LoanAmount), inplace=True)
df.Dependents.fillna(np.mean(df.Dependents), inplace=True)
df.Loan_Amount_Term.fillna(np.mean(df.Loan_Amount_Term), inplace=True)
df.Credit_History.fillna(0, inplace=True)
df.describe()


#df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')

X = df.iloc[:,1:12].values
y = df.iloc[:, 12].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
from sklearn.decomposition import PCA
pca = PCA(n_components = 11)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""
# Applying Kernel PCA
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim=11))
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
classifier.load_weights("loan_sanction_predict.h5")
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
#classifier.fit(X_train, y_train, batch_size = 1, epochs = 20000)
#classifier.save("loan_sanction_predict.h5")
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('mse', mean_squared_error(y_test, y_pred))
print('r2', r2_score(y_test, y_pred))
print('mae', mean_absolute_error(y_test, y_pred))
#Accuracy = 80%
#Accuracy using Random forest classifier 77.2
#Accuracy using LDA + Logistic regression 74%
#Accuracy using kernel svm with degree = 2, accuracy 79.2%
#Accuracy using Naive Bayes = 78.56%

x = ['male', 'no', 3, 'graduate', 'no', 1000, 2000, 100000, 360 , 0 , 'urban']
if(x[0] =="male" ):
 x[0] = 1
else :x[0] = 0

if(x[1] =="yes" ):
 x[1] = 1
else :x[1] = 0


if(x[3] =="graduate" ):
 x[3] = 1
else :x[3] = 0

if(x[4] =="no" ):
 x[4] = 0
else :x[4] = 1

if(x[10] =="urban" ):
 x[10] = 1
elif(x[10] =="semiurban" )  :x[10] = 2
else : x[10] = 0
x = np.asarray(x)
x = x.reshape(1,11)
y_pred_test = classifier.predict(x)
if y_pred_test == 1:
    print("Your loan will be approved")
else :
    print("sorry, improve your loan profile")
