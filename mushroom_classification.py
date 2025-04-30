# -*- coding: utf-8 -*-
"""Mushroom Classification Using Multi-Layer Perceptron
by Dana Shakrovsky

Original file is located at
    https://colab.research.google.com/drive/1-SDF5yKx-jN-6eJSzKHkfsuEoe6yVHJn
"""


""" Import packages and libraries """
# Pandas
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model

# Numpy
import numpy as np

# Plots
import matplotlib.pyplot as plt

# Import dataset
from ucimlrepo import fetch_ucirepo

# fetch dataset
secondary_mushroom = fetch_ucirepo(id=848)

# data (as pandas dataframes)
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

# metadata
print(secondary_mushroom.metadata)

# variable information
print(secondary_mushroom.variables)

""" Check Missing Values
Determine the percentage of NaN values in the columns for which UC Irvine indicated contain missing values. Depending on how much is missing, the columns will either be removed or will have the missing data imputed.
"""

# display data in its current state before modifications
print("Total number of samples: " + str(X.shape[0]))
X.head()

# determine percentage of NaN values in each column that contains missing values

# cap-surface column
cap_surface_col = X['cap-surface']
print(f"Number of NaN values in column 'cap-surface': {cap_surface_col.isna().sum().sum()}")
print("{:.2f}%\n".format((cap_surface_col.isna().sum().sum()/X.shape[0]) * 100))

# gill-attachment column
gill_attachment_col = X['gill-attachment']
print(f"Number of NaN values in column 'gill-attachment': {gill_attachment_col.isna().sum().sum()}")
print("{:.2f}%\n".format((gill_attachment_col.isna().sum().sum()/X.shape[0]) * 100))

# gill-spacing column
gill_spacing_col = X['gill-spacing']
print(f"Number of NaN values in column 'gill-spacing': {gill_spacing_col.isna().sum().sum()}")
print("{:.2f}%\n".format((gill_spacing_col.isna().sum().sum()/X.shape[0]) * 100))

# stem-root column
stem_root_col = X['stem-root']
print(f"Number of NaN values in column 'stem-root': {stem_root_col.isna().sum().sum()}")
print("{:.2f}%\n".format((stem_root_col.isna().sum().sum()/X.shape[0]) * 100))

# stem-surface column
stem_surface_col = X['stem-surface']
print(f"Number of NaN values in column 'stem-surface': {stem_surface_col.isna().sum().sum()}")
print("{:.2f}%\n".format((stem_surface_col.isna().sum().sum()/X.shape[0]) * 100))

# veil-color column
veil_color_col = X['veil-color']
print(f"Number of NaN values in column 'veil-color': {veil_color_col.isna().sum().sum()}")
print("{:.2f}%\n".format((veil_color_col.isna().sum().sum()/X.shape[0]) * 100))

# ring-type column
ring_type_col = X['ring-type']
print(f"Number of NaN values in column 'ring-type': {ring_type_col.isna().sum().sum()}")
print("{:.2f}%\n".format((ring_type_col.isna().sum().sum()/X.shape[0]) * 100))

# spore-print-color column
spore_print_color_col = X['spore-print-color']
print(f"Number of NaN values in column 'spore-print-color': {spore_print_color_col.isna().sum().sum()}")
print("{:.2f}%\n".format((spore_print_color_col.isna().sum().sum()/X.shape[0]) * 100))

# remove columns with greater than 20% of missing values
X_set = X.drop(['cap-surface', 'gill-spacing', 'stem-root', 'stem-surface','veil-color','spore-print-color'], axis=1)
X_set.head()

# impute missing data for columns with less than 20% missing values

# gill-attachment column
X_set['gill-attachment-n'] = pd.factorize(X_set['gill-attachment'], sort=True)[0] # first convert to numeric values
gill_attachment_n_col = ['gill-attachment-n']
for i in gill_attachment_n_col:
  X_set.loc[X_set.loc[:,i].isnull(), i]=X_set.loc[:,i].median() # replace NaN values with median

# ring-type column
X_set['ring-type-n'] = pd.factorize(X_set['ring-type'], sort=True)[0] # first convert to numeric values
ring_type_n_col = ['ring-type-n']
for i in ring_type_n_col:
  X_set.loc[X_set.loc[:,i].isnull(), i]=X_set.loc[:,i].median() # replace NaN values with median

# print results
X_clean = X_set.drop(['gill-attachment', 'ring-type'], axis=1)
print(f"Number of NaN values after imputation for column 'gill-attachment':\n{X_clean['gill-attachment-n'].isna().sum().sum()}")
print(f"\nNumber of NaN values after imputation for column 'ring-type':\n{X_clean['ring-type-n'].isna().sum().sum()}")

"""Convert categories into numeric values """

# cap-shape column
X_clean['cap-shape-n'] = pd.factorize(X_clean['cap-shape'], sort=True)[0]

# cap-color column
X_clean['cap-color-n'] = pd.factorize(X_clean['cap-color'], sort=True)[0]

# does-bruise-bleed column
X_clean['does-bruise-or-bleed-n'] = pd.factorize(X_clean['does-bruise-or-bleed'], sort=True)[0]

# gill-color column
X_clean['gill-color-n'] = pd.factorize(X_clean['gill-color'], sort=True)[0]

# stem-color column
X_clean['stem-color-n'] = pd.factorize(X_clean['stem-color'], sort=True)[0]

# veil-type column
X_clean['veil-type-n'] = pd.factorize(X_clean['veil-type'], sort=True)[0]

# has-ring  column
X_clean['has-ring-n'] = pd.factorize(X_clean['has-ring'], sort=True)[0]

# habitat column
X_clean['habitat-n'] = pd.factorize(X_clean['habitat'], sort=True)[0]

# season column
X_clean['season-n'] = pd.factorize(X_clean['season'], sort=True)[0]

# drop the original columns w/ categories
X_final = X_clean.drop(['cap-shape', 'cap-color', 'does-bruise-or-bleed', 'gill-color', 'stem-color','veil-type','has-ring','habitat','season'], axis=1)

X_final.head()

# convert target to binary numeric values
y_set = y.copy()
y_set['isPoisonous'] = pd.factorize(y['class'], sort=True)[0]
y_final = y_set.drop(['class'], axis=1)
y_final.head()

"""Split dataset into training and testing sets:
The ratio of training, validation, and test data is 80:10:10 (validation set will be split from the training set during training)
"""
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.1, random_state=0)

"""Create Multi-Layer Perceptron model"""

multi_perceptron = tf.keras.models.Sequential(layers=[
  # Input layer
  tf.keras.layers.Input(shape=(X_train.shape[1],)),

  # Inner layer
  tf.keras.layers.Dense(128, activation='sigmoid'),   # inner layer with 125 nodes, activation function = sigmoid

  # Output layer
  tf.keras.layers.Dense(1, activation='sigmoid')    # activation function = sigmoid
  ],
  trainable=True,
  name='Multi-layer_Perceptron'
)

# Compile
multi_perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

multi_perceptron.summary()

# Train the model
history = multi_perceptron.fit(
    x=X_train,
    y=y_train,
    epochs=10,
    batch_size=8,
    validation_split=0.1,
    verbose=2,
)

""" Test the Multi-Layer Perceptron model's accuracy:
Test the multi-layer perceptron with the test set, then evaluate the results with f1, accuracy, precision, and recall scores.
"""

# Test perceptron with test dataset
loss = multi_perceptron.evaluate(X_test, y_test)

# get predictions
y_predictions = multi_perceptron.predict(X_test)

# convert predictions into categories
y_class_pred = []
for y_pred in y_predictions:
  # conversion rule: y_pred = 1 iff probability of category 1 ≥ 0.5
  if y_pred >= 0.5:
    y_class_pred = np.append(y_class_pred, 1)
  else:
    y_class_pred = np.append(y_class_pred, 0)

# calculate f1 score and accuracy
f1 = f1_score(y_test, y_class_pred)
accuracy = accuracy_score(y_test, y_class_pred)

# calculate precision and recall
precision = precision_score(y_test, y_class_pred)
recall = recall_score(y_test, y_class_pred)

"""Confusion Matrix for Multi-Layer Perceptron"""

# create confusion matrix
conf_matrix = confusion_matrix(y_test, y_class_pred)
confusion_matrix_diagram = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])

confusion_matrix_diagram.plot()
plt.show()

# print training, validation, and test losses
print("\nSummary of Multi-Layer Perceptron Results:")
print("\nTraining loss: " + str(np.array(history.history['loss']).mean()))
print("Validation loss: " + str(np.array(history.history['val_loss']).mean()))
print("Test loss: " + str(np.array(loss).mean()))

# print f1, accuracy, precision, and recall scores
print("\nf1 score: " + str(f1))
print("accuracy score: " + str(accuracy))
print("precision score: " + str(precision))
print("recall score: " + str(recall))

"""Compare with Single-Layer Perceptron Model"""

single_perceptron = tf.keras.models.Sequential(layers=[
  # Input layer
  tf.keras.layers.Input(shape=(X_train.shape[1],)),

  # Output layer
  tf.keras.layers.Dense(1, activation='sigmoid')    # activation function = sigmoid
  ],
  trainable=True,
  name='Single-layer_Perceptron'
)

# Compile
single_perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

single_perceptron.summary()

# Train the model
history2 = single_perceptron.fit(
    x=X_train,
    y=y_train,
    epochs=10,
    batch_size=8,
    validation_split=0.1,
    verbose=2,
)

# Evaluate

# Test perceptron with test dataset
loss2 = single_perceptron.evaluate(X_test, y_test)

# get predictions
y_predictions2 = single_perceptron.predict(X_test)

# convert predictions into categories
y_class_pred2 = []
for y_pred2 in y_predictions2:
  # conversion rule: y_pred = 1 iff probability of category 1 ≥ 0.5
  if y_pred2 >= 0.5:
    y_class_pred2 = np.append(y_class_pred2, 1)
  else:
    y_class_pred2 = np.append(y_class_pred2, 0)

# calculate f1 score and accuracy
f1_2 = f1_score(y_test, y_class_pred2)
accuracy_2 = accuracy_score(y_test, y_class_pred2)

# calculate precision and recall
precision_2 = precision_score(y_test, y_class_pred2)
recall_2 = recall_score(y_test, y_class_pred2)

# Confusion Matrix
conf_matrix2 = confusion_matrix(y_test, y_class_pred2)
confusion_matrix_diagram2 = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix2, display_labels = [False, True])

confusion_matrix_diagram2.plot()
plt.show()

# print training, validation, and test losses
print("\nSummary of Single-Layer Perceptron Results:")
print("\nTraining loss: " + str(np.array(history2.history['loss']).mean()))
print("Validation loss: " + str(np.array(history2.history['val_loss']).mean()))
print("Test loss: " + str(np.array(loss2).mean()))

# print f1, accuracy, precision, and recall scores
print("\nf1 score: " + str(f1_2))
print("accuracy score: " + str(accuracy_2))
print("precision score: " + str(precision_2))
print("recall score: " + str(recall_2))
