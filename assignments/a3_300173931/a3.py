#1, Load the Dataset
import pandas as pd

# URLs for the datasets (replace with actual URLs if these are not correct)
train_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a3/cb513_train.csv'
validation_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a3/cb513_valid.csv'
test_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a3/cb513_test.csv'

# Load the datasets
train_data = pd.read_csv(train_url, header=None)
validation_data = pd.read_csv(validation_url, header=None)
test_data = pd.read_csv(test_url, header=None)

# Display basic information about each dataset
print("Training Data:")
print(train_data.info())
print(train_data.head())

print("\nValidation Data:")
print(validation_data.info())
print(validation_data.head())

print("\nTest Data:")
print(test_data.info())
print(test_data.head())

#2, Shuffling the Rows
from sklearn.utils import shuffle

# Shuffle the rows of each dataset
train_data = shuffle(train_data, random_state=42)  # Fixing random state for reproducibility
validation_data = shuffle(validation_data, random_state=42)
test_data = shuffle(test_data, random_state=42)

# Display the first few rows of each dataset to confirm shuffling
print("Shuffled Training Data:")
print(train_data.head())

print("\nShuffled Validation Data:")
print(validation_data.head())

print("\nShuffled Test Data:")
print(test_data.head())


#3, Scaling of Numerical Features
from sklearn.preprocessing import MinMaxScaler

# Separate features and target
# Assuming the target column is the first one (adjust if necessary)
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_validation = validation_data.iloc[:, 1:]
y_validation = validation_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Option 1: Without scaling
# Use X_train, X_validation, and X_test directly for training and evaluation

# Option 2: With MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# You now have two sets of feature matrices to test:
# 1. X_train, X_validation, X_test - without scaling
# 2. X_train_scaled, X_validation_scaled, X_test_scaled - with MinMax scaling

# Display a summary to confirm scaling
print("Without scaling (first 5 rows):")
print(X_train.head())

print("\nWith MinMax scaling (first 5 rows):")
print(X_train_scaled[:5])

#4, Isolating the Target and the Data
# For the training dataset
X_train = train_data.iloc[:, 1:]  # Select all columns except the first
y_train = train_data.iloc[:, 0]   # Select the first column as target

# For the validation dataset
X_validation = validation_data.iloc[:, 1:]
y_validation = validation_data.iloc[:, 0]

# For the test dataset
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Display a quick check of the shapes to confirm separation
print("Training Data (X_train):", X_train.shape)
print("Training Target (y_train):", y_train.shape)

print("\nValidation Data (X_validation):", X_validation.shape)
print("Validation Target (y_validation):", y_validation.shape)

print("\nTest Data (X_test):", X_test.shape)
print("Test Target (y_test):", y_test.shape)

#5, Model Development
#Dummy Model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Initialize the DummyClassifier to always predict the majority class
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# Evaluate the Dummy model on the validation set
dummy_predictions = dummy_model.predict(X_validation)
dummy_accuracy = accuracy_score(y_validation, dummy_predictions)

print("Dummy Model Accuracy:", dummy_accuracy)

#Baseline Model
from sklearn.linear_model import LogisticRegression

# Initialize and train the baseline model (Logistic Regression)
baseline_model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
baseline_model.fit(X_train, y_train)

# Evaluate the baseline model on the validation set
baseline_predictions = baseline_model.predict(X_validation)
baseline_accuracy = accuracy_score(y_validation, baseline_predictions)

print("Baseline Model (Logistic Regression) Accuracy:", baseline_accuracy)

#Neural Network Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Define the neural network model
nn_model = Sequential([
    Dense(462, input_dim=462, activation='relu'),  # Input layer
    Dense(8, activation='relu'),                   # Hidden layer
    Dense(3, activation='softmax')                 # Output layer with softmax for probabilities
])

# Compile the model
nn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training set
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_validation, y_validation))

# Evaluate the neural network on the validation set
nn_loss, nn_accuracy = nn_model.evaluate(X_validation, y_validation, verbose=0)
print("Neural Network Model Accuracy:", nn_accuracy)

#6,  Model Evaluation
#Cross-Validation for Baseline Model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Define scoring metrics
scoring = {
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Perform 5-fold cross-validation on the baseline model
baseline_cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring='f1_weighted')
print("Baseline Model - Cross-validated F1 Score:", baseline_cv_scores.mean())

#Evaluation of the Neural Network Using Validation Set
from sklearn.metrics import precision_score, recall_score, f1_score

# Predict on the validation set
nn_validation_predictions = nn_model.predict(X_validation)
nn_validation_predictions = nn_validation_predictions.argmax(axis=1)  # Convert probabilities to class predictions

# Calculate precision, recall, and F1-score for the neural network
nn_precision = precision_score(y_validation, nn_validation_predictions, average='weighted')
nn_recall = recall_score(y_validation, nn_validation_predictions, average='weighted')
nn_f1 = f1_score(y_validation, nn_validation_predictions, average='weighted')

print("Neural Network Model - Validation Precision:", nn_precision)
print("Neural Network Model - Validation Recall:", nn_recall)
print("Neural Network Model - Validation F1 Score:", nn_f1)

#Evaluation of Dummy Model
# Calculate precision, recall, and F1-score for the dummy model
dummy_precision = precision_score(y_validation, dummy_predictions, average='weighted')
dummy_recall = recall_score(y_validation, dummy_predictions, average='weighted')
dummy_f1 = f1_score(y_validation, dummy_predictions, average='weighted')

print("Dummy Model - Validation Precision:", dummy_precision)
print("Dummy Model - Validation Recall:", dummy_recall)
print("Dummy Model - Validation F1 Score:", dummy_f1)

#7, Baseline Model:
#Decision Tree Classifier Hyperparameter Tuning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Decision Tree
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None]  # Testing a range of depths, including no limit
}

# Initialize GridSearchCV for Decision Tree
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
dt_grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best parameters for Decision Tree:", dt_grid_search.best_params_)
print("Best F1 Score for Decision Tree:", dt_grid_search.best_score_)

#Logistic Regression Hyperparameter Tuning
from sklearn.linear_model import LogisticRegression

# Define the parameter grid for Logistic Regression
lr_param_grid = {
    'penalty': ['l2', 'none'],
    'max_iter': [100, 500, 1000],
    'tol': [1e-4, 1e-3, 1e-2]
}

# Initialize GridSearchCV for Logistic Regression
lr_grid_search = GridSearchCV(LogisticRegression(solver='saga', multi_class='multinomial'), lr_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
lr_grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best parameters for Logistic Regression:", lr_grid_search.best_params_)
print("Best F1 Score for Logistic Regression:", lr_grid_search.best_score_)

#K-Nearest Neighbors Hyperparameter Tuning
from sklearn.neighbors import KNeighborsClassifier

# Define the parameter grid for K-Nearest Neighbors
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# Initialize GridSearchCV for KNN
knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
knn_grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best parameters for K-Nearest Neighbors:", knn_grid_search.best_params_)
print("Best F1 Score for K-Nearest Neighbors:", knn_grid_search.best_score_)

#8, Neural Network:
#Single Hidden Layer - Varying Number of Nodes
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def train_and_plot_model(nodes, epochs=50):
    # Build the model with a single hidden layer
    model = Sequential([
        Dense(462, input_dim=462, activation='relu'),  # Input layer
        Dense(nodes, activation='relu'),               # Hidden layer with variable nodes
        Dense(3, activation='softmax')                 # Output layer
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model and capture history
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_validation, y_validation), verbose=0)
    
    # Plot loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'Training and Validation Loss (Hidden Nodes: {nodes})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'Training and Validation Accuracy (Hidden Nodes: {nodes})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.show()

# Example with 1 node in hidden layer
train_and_plot_model(nodes=1)

#Varying Number of Layers
def train_and_plot_multilayer_model(layers=2, nodes_per_layer=[256, 128], epochs=50):
    model = Sequential()
    model.add(Dense(462, input_dim=462, activation='relu'))  # Input layer
    
    # Add hidden layers with specified nodes
    for nodes in nodes_per_layer:
        model.add(Dense(nodes, activation='relu'))
    
    model.add(Dense(3, activation='softmax'))  # Output layer
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_validation, y_validation), verbose=0)
    
    # Plotting code same as above
    # ...

# Example with 2 layers in pyramid structure
train_and_plot_multilayer_model(layers=2, nodes_per_layer=[256, 128])

#Regularization Techniques
from tensorflow.keras.regularizers import l2

def train_with_l2_regularization(nodes=128, l2_value=0.01):
    model = Sequential([
        Dense(462, input_dim=462, activation='relu', kernel_regularizer=l2(l2_value)),
        Dense(nodes, activation='relu', kernel_regularizer=l2(l2_value)),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation), verbose=0)
    
    # Plotting code same as above
    # ...

# Experiment with different L2 values
train_with_l2_regularization(nodes=128, l2_value=0.01)
train_with_l2_regularization(nodes=128, l2_value=0.001)
train_with_l2_regularization(nodes=128, l2_value=1e-4)

#Dropout Layers
#Dropout layers with dropout rates of 0.25 and 0.5
from tensorflow.keras.layers import Dropout

def train_with_dropout(nodes=128, dropout_rate=0.25):
    model = Sequential([
        Dense(462, input_dim=462, activation='relu'),
        Dropout(dropout_rate),  # Add dropout layer
        Dense(nodes, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation), verbose=0)
    
    # Plotting code same as above
    # ...

# Experiment with different dropout rates
train_with_dropout(nodes=128, dropout_rate=0.25)
train_with_dropout(nodes=128, dropout_rate=0.5)

#Early stopping
from tensorflow.keras.callbacks import EarlyStopping

def train_with_early_stopping(nodes=128, patience=5):
    model = Sequential([
        Dense(462, input_dim=462, activation='relu'),
        Dense(nodes, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation), verbose=0, callbacks=[early_stopping])
    
    # Plotting code same as above
    # ...

# Run training with early stopping
train_with_early_stopping(nodes=128, patience=5)

#9, Model Comparison
#Evaluate the Baseline Model on the Test Set
from sklearn.metrics import precision_score, recall_score, f1_score

# Make predictions with the best baseline model on the test set
baseline_best_model = dt_grid_search.best_estimator_  # Assuming Decision Tree was optimal; update as needed
baseline_test_predictions = baseline_best_model.predict(X_test)

# Calculate precision, recall, and F1-score for the baseline model
baseline_precision = precision_score(y_test, baseline_test_predictions, average='weighted')
baseline_recall = recall_score(y_test, baseline_test_predictions, average='weighted')
baseline_f1 = f1_score(y_test, baseline_test_predictions, average='weighted')

print("Baseline Model - Test Precision:", baseline_precision)
print("Baseline Model - Test Recall:", baseline_recall)
print("Baseline Model - Test F1 Score:", baseline_f1)

#Evaluate the Best Neural Network Configuration on the Test Set
# Make predictions with the best neural network model on the test set
nn_test_predictions = nn_model.predict(X_test)
nn_test_predictions = nn_test_predictions.argmax(axis=1)  # Convert probabilities to class predictions

# Calculate precision, recall, and F1-score for the neural network model
nn_precision = precision_score(y_test, nn_test_predictions, average='weighted')
nn_recall = recall_score(y_test, nn_test_predictions, average='weighted')
nn_f1 = f1_score(y_test, nn_test_predictions, average='weighted')

print("Neural Network Model - Test Precision:", nn_precision)
print("Neural Network Model - Test Recall:", nn_recall)
print("Neural Network Model - Test F1 Score:", nn_f1)

#Dummy Model Benchmark on the Test Set
# Make predictions with the dummy model on the test set
dummy_test_predictions = dummy_model.predict(X_test)

# Calculate precision, recall, and F1-score for the dummy model
dummy_precision = precision_score(y_test, dummy_test_predictions, average='weighted')
dummy_recall = recall_score(y_test, dummy_test_predictions, average='weighted')
dummy_f1 = f1_score(y_test, dummy_test_predictions, average='weighted')

print("Dummy Model - Test Precision:", dummy_precision)
print("Dummy Model - Test Recall:", dummy_recall)
print("Dummy Model - Test F1 Score:", dummy_f1)