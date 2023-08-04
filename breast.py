import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('data.csv')
data.head()
"""
569 rows x 33 columns

	id	diagnosis	radius_mean	texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave points_mean	...	texture_worst	perimeter_worst	area_worst	smoothness_worst	compactness_worst	concavity_worst	concave points_worst	symmetry_worst	fractal_dimension_worst	Unnamed: 32
0	842302	M	17.99	10.38	122.80	1001.0	0.11840	0.27760	0.3001	0.14710	...	17.33	184.60	2019.0	0.1622	0.6656	0.7119	0.2654	0.4601	0.11890	NaN
1	842517	M	20.57	17.77	132.90	1326.0	0.08474	0.07864	0.0869	0.07017	...	23.41	158.80	1956.0	0.1238	0.1866	0.2416	0.1860	0.2750	0.08902	NaN
2	84300903	M	19.69	21.25	130.00	1203.0	0.10960	0.15990	0.1974	0.12790	...	25.53	152.50	1709.0	0.1444	0.4245	0.4504	0.2430	0.3613	0.08758	NaN
3	84348301	M	11.42	20.38	77.58	386.1	0.14250	0.28390	0.2414	0.10520	...	26.50	98.87	567.7	0.2098	0.8663	0.6869	0.2575	0.6638	0.17300	NaN
4	84358402	M	20.29	14.34	135.10	1297.0	0.10030	0.13280	0.1980	0.10430	...	16.67	152.20	1575.0	0.1374	0.2050	0.4000	0.1625	0.2364	0.07678	NaN

"""
# Get the column names as a list
data.columns
"""
Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'],
      dtype='object')
"""

def get_column_values_as_list(df, column_name):
  column_values = df[column_name].tolist()
  return column_values
diagnosis_values = get_column_values_as_list(data, "diagnosis")

import collections
counter = collections.Counter(diagnosis_values)
print(counter)
"""
Counter({'B': 357, 'M': 212})
"""


data.isnull().sum()

X = data[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

X.shape
# (569, 10)

label_map = {"B": 0, "M": 1}
# Convert the labels to numbers.
y = [label_map[label] for label in data['diagnosis']]

y = np.array(y)
y.shape
# (569, )

# Split the data into a training set and a test set.
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)
score = model.score(x_test, y_test)

print(f"The Accuracy of LOGISTIC REGRESSION Model: {score*100 }%")

# Cross-validated accuracy
cv_rf = cross_val_score(model,
                       X,
                       y,
                       cv=10,
                       scoring='accuracy')
cv_acc = np.mean(cv_rf)




# Cross-validated precision
cv_precision = cross_val_score(model,
                       X,
                       y,
                       cv=10,
                       scoring='precision')
cv_precision = np.mean(cv_precision)


# Cross-validated recall
cv_recall = cross_val_score(model,
                       X,
                       y,
                       cv=10,
                       scoring='recall')
cv_recall = np.mean(cv_recall)




print(f"Cross-validated accuracy: {cv_acc * 100}%")
print(f"Cross-validated precision: {cv_precision*100}%")
print(f"Cross-validated recall: {cv_recall*100}%")


"""
The Accuracy of LOGISTIC REGRESSION Model: 92.10526315789474%
Cross-validated accuracy: 90.85526315789474%
Cross-validated precision: 89.97940746044462%
Cross-validated recall: 85.4978354978355%
"""