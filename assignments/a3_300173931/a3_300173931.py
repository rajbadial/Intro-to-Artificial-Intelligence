import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

#1, Load the Dataset

# URLs for the datasets (replace with actual URLs if these are not correct)
train_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a3/cb513_train.csv'
validation_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a3/cb513_valid.csv'
test_url = 'https://raw.githubusercontent.com/turcotte/csi4106-f24/refs/heads/main/assignments-data/a3/cb513_test.csv'

# Load the datasets
train_data = pd.read_csv(train_url, header=None)
validation_data = pd.read_csv(validation_url, header=None)
test_data = pd.read_csv(test_url, header=None)

shuffled_training = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display basic information about each dataset
""" print("Training Data:")
print(train_data.head())

print("\nValidation Data:")
print(validation_data.head())

print("\nTest Data:")
print(test_data.head()) """

# Step 3: Separating the data
y = shuffled_training.iloc[:, 0]    # target vector
X = shuffled_training.iloc[:, 1:]   # features

# Assuming X and y are defined as the features and target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Option 1: Without scaling
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)
no_scaling_score = model.score(X_val, y_val)

# Option 2: With MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model.fit(X_train_scaled, y_train)
scaling_score = model.score(X_val_scaled, y_val)

# Compare results
print("Score without scaling:", no_scaling_score)
print("Score with MinMaxScaler:", scaling_score)

# Step 4: Isolating the data for each dataset (already done for training in step 3, labeled just as y and x)
y_validation = validation_data.iloc[:, 0]    # target vector
X_validation = validation_data.iloc[:, 1:]   # features

y_test = test_data.iloc[:, 0]    # target vector
X_test = test_data.iloc[:, 1:]   # features

# Step 5
# Implementing Dummy model
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)
y_val_pred_dummy = dummy_model.predict(X_val)

# Implementing Baseline model (with DecisionTree)
baseline_model = DecisionTreeClassifier(max_depth=10, random_state=42)
baseline_model.fit(X_train, y_train)
y_val_pred_baseline = baseline_model.predict(X_val)

# Using decision tree as many of the relationships are likely non-linear which decision tree is good at predicting, 
# while Logistic Regression is better for linear predictions. And KNN is better for smaller datasets where the data naturally 
# clusters around each class. Because of high-dimensionality KNN may not be the best fit. Decision Tree is prone to overfitting however.

# Implementing Neural Network Model
def create_neural_network(hidden_nodes):
    model = Sequential()
    model.add(Input(shape=(462,)))                                # Input layer with 462 nodes
    model.add(Dense(hidden_nodes, activation='relu'))             # Hidden layer with 8 nodes (using hidden_nodes to reuse this function later for step 8)
    model.add(Dense(3, activation='softmax'))                     # Output layer with 3 nodes
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_neural_network(8)
history = nn_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val)) # train the model
y_val_pred_nn = nn_model.predict(X_val).argmax(axis=1) # because we're using softmax in the NNM

# Step 6: Cross-Eval of Models
# Dummy Model
accuracy_dummy = accuracy_score(y_val, y_val_pred_dummy)
precision_dummy = precision_score(y_val, y_val_pred_dummy, average='weighted', zero_division=1)
recall_dummy = recall_score(y_val, y_val_pred_dummy, average='weighted')
f1_dummy = f1_score(y_val, y_val_pred_dummy, average='weighted')

print("\nDummy Model Performance:")
print(f"Accuracy: {accuracy_dummy:.2f}")
print(f"Precision: {precision_dummy:.2f}")
print(f"Recall: {recall_dummy:.2f}")
print(f"F1 Score: {f1_dummy:.2f}")

# Baseline Model w 5 folds
scoring = {
    'accuracy': make_scorer(accuracy_score), 
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Perform cross-validation
cv_results = cross_validate(baseline_model, X_train, y_train, cv=5, scoring=scoring)

# Display average cross-validation scores
print("\nBaseline Model Performance:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Decision Tree CV Average Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Decision Tree CV Average Recall: {cv_results['test_recall'].mean():.4f}")
print(f"Decision Tree CV Average F1-Score: {cv_results['test_f1'].mean():.4f}")

# NN Model
accuracy_nn = accuracy_score(y_val, y_val_pred_nn)
precision_nn = precision_score(y_val, y_val_pred_nn, average='weighted')
recall_nn = recall_score(y_val, y_val_pred_nn, average='weighted')
f1_nn = f1_score(y_val, y_val_pred_nn, average='weighted')

print("\nNeural Network Model Performance:")
print(f"Accuracy: {accuracy_nn:.2f}")
print(f"Precision: {precision_nn:.2f}")
print(f"Recall: {recall_nn:.2f}")
print(f"F1 Score: {f1_nn:.2f}")

# Step 7: Hyperparameter Optimization for Baseline model
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],   # Limits depth to avoid overfitting
    'min_samples_split': [2, 5, 10]    # Controls when a node will be split
}

scoring = {
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Initialize GridSearchCV with the Decision Tree model and 5-fold cross-validation
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring=scoring,
    refit='f1',                      # Refits on the best parameter setting based on F1 score
    cv=5,
    verbose=1,
    n_jobs=-1                        # Use all available CPU cores
)

grid_search.fit(X_train, y_train)

# best parameters found
print("Best parameters found:", grid_search.best_params_)

# average precision and recall scores for the best model
best_cv_results = grid_search.cv_results_
best_index = grid_search.best_index_
print("Best Cross-Validated Precision:", best_cv_results['mean_test_precision'][best_index])
print("Best Cross-Validated Recall:", best_cv_results['mean_test_recall'][best_index])
print("Best Cross-Validated F1 Score:", best_cv_results['mean_test_f1'][best_index])

# Step 8 (Neural Network Experimentation)
# Varying nodes in hidden layer:

# List of node counts to try in the hidden layer
node_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Dictionary to store the validation accuracy for each configuration
results = {}

# Loop through each node configuration, train the model, and record the validation accuracy
for nodes in node_counts:
    print(f"\nTraining model with {nodes} hidden nodes...")
    nn_model = create_neural_network(hidden_nodes=nodes) # Same function from Step 5
    history = nn_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
    
    # Plot training and validation loss and accuracy
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title(f'Loss for {nodes} Nodes in Hidden Layer')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'Accuracy for {nodes} Nodes in Hidden Layer')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Varying number of layers:
def create_neural_network2(num_layers, nodes_per_layer=8):
    model = Sequential()
    model.add(Input(shape=(462,)))  # Input layer with 462 features
    
    # Add the specified number of hidden layers with the same number of nodes per layer
    for _ in range(num_layers):
        model.add(Dense(nodes_per_layer, activation='relu'))
    
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 nodes for 3 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List of layer counts to try
layer_counts = [1, 2, 3, 4, 5]

# Loop through each configuration, train the model, and plot the results
for layers in layer_counts:
    print(f"\nTraining model with {layers} hidden layers")
    nn_model = create_neural_network2(num_layers=layers)
    history = nn_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
    
    # Plot training and validation loss and accuracy
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title(f'Loss for {layers} Hidden Layers')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'Accuracy for {layers} Hidden Layers')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Changing the activation function
def create_neural_network3(activation='relu', num_layers=2, nodes_per_layer=8):
    model = Sequential()
    model.add(Input(shape=(462,)))  # Input layer with 462 features
    
    # Add hidden layers with the specified activation function
    for _ in range(num_layers):
        model.add(Dense(nodes_per_layer, activation=activation))
    
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 nodes for 3 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List of activation functions to try
activation_functions = ['relu', 'sigmoid']

# Loop through each activation function, train the model, and plot the results
for activation in activation_functions:
    print(f"\nTraining model with {activation} activation function")
    nn_model = create_neural_network3(activation=activation)
    history = nn_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
    
    # Plot training and validation loss and accuracy
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title(f'Loss with {activation} Activation Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'Accuracy with {activation} Activation Function')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Regularization techniques
def create_neural_network4(l2_penalty=0.0, dropout_rate=0.0):
    model = Sequential()
    model.add(Input(shape=(462,)))
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(l2_penalty)))  # L2 regularization on the hidden layer
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))  # Apply dropout if rate > 0
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define different regularization configurations to try
regularization_configs = [
    {'l2_penalty': 0.01, 'dropout_rate': 0.0}, # Only L2 regularization
    {'l2_penalty': 0.001, 'dropout_rate': 0.0}, # Smaller L2 regularization
    {'l2_penalty': 0.01, 'dropout_rate': 0.25}, # Both L2 and Dropout of 0.25
    {'l2_penalty': 0.001, 'dropout_rate': 0.25}, # Smaller L2 with 0.25 dropoyt
    {'l2_penalty': 0.01, 'dropout_rate': 0.5}, # L2 with 0.5 dropout
    {'l2_penalty': 0.001, 'dropout_rate': 0.5}, # Smaller L2 with 0.5 dropout 
    {'l2_penalty': 0.0, 'dropout_rate': 0.25}, # Only Dropout of 0.25
    {'l2_penalty': 0.0, 'dropout_rate': 0.5} # Only Dropout of 0.5
]

# Loop through each regularization configuration
for config in regularization_configs:
    l2_penalty = config['l2_penalty']
    dropout_rate = config['dropout_rate']
    print(f"\nTraining model with L2 penalty={l2_penalty} and Dropout rate={dropout_rate}")
    
    nn_model = create_neural_network4(l2_penalty=l2_penalty, dropout_rate=dropout_rate)
    history = nn_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
    
    # Plot training and validation loss and accuracy
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title(f'Loss with L2={l2_penalty} and Dropout={dropout_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'Accuracy with L2={l2_penalty} and Dropout={dropout_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
# Step 9

# Decision Tree model with specified hyperparameters
decision_tree_model = DecisionTreeClassifier(criterion='entropy', min_samples_split=5, max_depth=10)

# Train the Decision Tree model
decision_tree_model.fit(X_train, y_train)

# Predict on the validation/test set
y_pred_dt = decision_tree_model.predict(X_test)

# Calculate evaluation metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted', zero_division=1)
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

# Print metrics
print("Decision Tree Model Evaluation")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1 Score:", f1_dt)

# Define the neural network structure
def create_neural_network5(num_layers, l2_penalty, dropout_rate, nodes_per_layer):
    model = Sequential()
    model.add(Input(shape=(462,)))
    for _ in range(num_layers):
        model.add(Dense(nodes_per_layer, activation='relu', kernel_regularizer=l2(l2_penalty)))  # L2 regularization on the hidden layer
        model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

neural_network_model = create_neural_network5(l2_penalty=0.01, dropout_rate=0.25, nodes_per_layer=8, num_layers=2)

# Train the model (adjust epochs and batch size as needed)
history = neural_network_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Predict and evaluate the model
y_pred_nn = neural_network_model.predict(X_test).argmax(axis=1)

# Calculate evaluation metrics
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn, average="weighted", zero_division=1)
recall_nn = recall_score(y_test, y_pred_nn, average='weighted')
f1_nn = f1_score(y_test, y_pred_nn, average='weighted')

# Print metrics
print("Neural Network Model Evaluation")
print("Accuracy:", accuracy_nn)
print("Precision:", precision_nn)
print("Recall:", recall_nn)
print("F1 Score:", f1_nn)


# Resources
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
# https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html 
# https://www.geeksforgeeks.org/implementing-neural-networks-using-tensorflow/
# https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html

"""
AI Code given when asked how to display loss and accuracy for each set of nodes in step 8 part 1 (Used mainly to save time and to keep code clean)
for nodes in node_counts:
    print(f"\nTraining model with {nodes} hidden nodes...")
    nn_model = create_neural_network(hidden_nodes=nodes)
    history = nn_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    
    # Plot training and validation loss and accuracy
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title(f'Loss for {nodes} Nodes in Hidden Layer')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'Accuracy for {nodes} Nodes in Hidden Layer')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
"""