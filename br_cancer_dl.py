import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin

# Import the data
data = pd.read_csv('/data.csv')
data = data.drop(columns=['Unnamed: 32'])  # Drop the last column

# Preprocessing
scaler = StandardScaler()
X = data[['radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
y = data['diagnosis'].map({'M': 1, 'B': 0})

X = scaler.fit_transform(X)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# data ->  tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# DL Model class
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

### 
###
model = Net(X_train.shape[1])
###
###

# loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 100
batch_size = 16
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(num_batches):
        # batch
        batch_indices = torch.randperm(len(X_train))[:batch_size]
        batch_X = X_train[batch_indices]
        batch_y = y_train[batch_indices]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)

        # Backward pass & opt.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / num_batches}")

# Evaluation
with torch.no_grad():
    predictions = model(X_test)
    predictions = torch.round(predictions).squeeze()
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100}%")


"""
Epoch 1/100, Loss: 0.13237052666240093
Epoch 2/100, Loss: 0.07869710172053601
Epoch 3/100, Loss: 0.16327773041224905
Epoch 4/100, Loss: 0.0460232486098643
Epoch 5/100, Loss: 0.12600912562733615
Epoch 6/100, Loss: 0.05392403676939596
Epoch 7/100, Loss: 0.14546180561436423
Epoch 8/100, Loss: 0.0892275398197983
Epoch 9/100, Loss: 0.12963454285188294
Epoch 10/100, Loss: 0.12933745579461434
Epoch 11/100, Loss: 0.15609584901747958
Epoch 12/100, Loss: 0.07327102727556069
Epoch 13/100, Loss: 0.09659222385380417
Epoch 14/100, Loss: 0.09317193925380707
Epoch 15/100, Loss: 0.06993880290870688
Epoch 16/100, Loss: 0.11733589714068719
Epoch 17/100, Loss: 0.10817687100331698
Epoch 18/100, Loss: 0.059232932228561755
Epoch 19/100, Loss: 0.10994474401897085
Epoch 20/100, Loss: 0.11813556258234062
Epoch 21/100, Loss: 0.11047529559956663
Epoch 22/100, Loss: 0.036373746832915846
Epoch 23/100, Loss: 0.11187087511539826
Epoch 24/100, Loss: 0.12391349221122384
Epoch 25/100, Loss: 0.09192312024866364...
Epoch 98/100, Loss: 0.027507446016118462
Epoch 99/100, Loss: 0.11977237235779674
Epoch 100/100, Loss: 0.14886602932321175
Accuracy: 97.36842105263158%
"""