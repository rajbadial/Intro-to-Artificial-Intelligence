import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

url1 = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/01/glass.csv'
url2 = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/02/dermatology_database_1.csv'
url3 = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/03/Maternal%20Health%20Risk%20Data%20Set.csv'
# For URL 4, where the column headers are separated
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class values']
df = pd.read_csv('https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/04/car.data', header=None)
# Assign the correct column headers to the DataFrame
df.columns = columns
url5 = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/05/WineQT.csv'
url6 = 'https://github.com/turcotte/csi4106-f24/raw/refs/heads/main/assignments-data/a1/06/16P.csv'
url7 = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a1/07/train.csv'
#df = pd.read_csv(url6, encoding='ISO-8859-1')
df = pd.read_csv(url5)

# Display unique values for each column
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("\n")


print(df.describe(include='all'))
print(df.isnull().sum())

df = df.drop(columns='Id')

# Drop 'quality' from the features
X = df.drop(columns=['quality'])

# Target variable is 'quality'
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 2: Print the shapes of the resulting datasets
print("Training Feature Set (X_train) Shape:", X_train.shape)
print("Training Target Set (y_train) Shape:", y_train.shape)
print("Test Feature Set (X_test) Shape:", X_test.shape)
print("Test Target Set (y_test) Shape:", y_test.shape)