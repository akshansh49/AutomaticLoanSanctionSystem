import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('train_v2.csv')
df.head()
df.describe()
df.isnull().values.any()
df = df.replace('NA', np.nan)
is_numeric = np.vectorize(lambda x: np.issubdtype(x, np.number))
mask_is_numeric = is_numeric(df.dtypes)
np.all(mask_is_numeric)
df.loc[:, ~mask_is_numeric] = df.loc[:, ~mask_is_numeric].applymap(float)
X = df.loc[:,'f1':'f778'].values
y = df.loc[:,'loss'].values
imp = Imputer(strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)
np.any(np.isnan(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('mse', mean_squared_error(y_test, y_pred))
print('r2', r2_score(y_test, y_pred))
print('mae', mean_absolute_error(y_test, y_pred))

df_test = pd.read_csv('test_v2.csv')
df_test = df_test.replace('NA', np.nan)
mask_is_numeric = is_numeric(df_test.dtypes)
df_test.loc[:, ~mask_is_numeric] = df_test.loc[:, ~mask_is_numeric].applymap(float)
X_sub = df_test.loc[:,'f1':'f778'].values
imp = Imputer(strategy='mean', axis=0)
imp.fit(X_sub)
X_sub = imp.transform(X_sub)

df_test['loss'] = clf.predict(X_sub)
print("The total net transaction value of bank is %f"%(sum(df_test['loss'])))

good =0
bad = 0
for i in df_test['loss']:
    if i>(-10000):
        good+=1
    else :bad+=1
print("The number of defaulters are %d \n"%(bad))
print("The number of healthy seads are %d \n"%(good))