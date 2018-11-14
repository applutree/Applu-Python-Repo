# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#-----------------------------------Data Visualization/Exploration-----------------------------------#

# Show the dimensions of the dataset
# In this case it shows (150, 5) which means 150 instances and 5 attributes
print(dataset.shape)

# Look at the first 20 rows of the data
print(dataset.head(20))

# Statistical Summary - descriptive - Quick Overview of count, mean, std, min, max
print(dataset.describe())

# Get the count of each class
print(dataset.groupby('class').size())

# Box and whisker plots if each input attributes
# Allows a quick view of the distribution
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Histograms - Check for type of distribution
# Depending on the type of distribution - each have their own properties to be aware of
dataset.hist()
plt.show()

# Scatter plot matrix
# Note the diagonal grouping of some pairs of attributes. 
# This suggests a high correlation and a predictable relationship.
scatter_matrix(dataset)
plt.show()

#-----------------------------------Data Visualization/Exploration End-------------------------------#

# Split-out validation dataset
# 1 - Create 'array' which is a list of dataset with values only
array = dataset.values
# print(array)

# 2 - Split list using : (slicing)
# Notation for [rows, columns]
# We are selecting columns using array slicing in Python using ranges.

# X is comprised of columns 0, 1, 2 and 3.
X = array[:,0:4]
# print(X)

# Y is comprised of column 4.
Y = array[:,4]
# print(Y)

# Splitting 20% of the data for validation set
validation_size = 0.20

# Set seed to 7 - Seed defines a starting point for the generation of random values. 
# Running an analysis with the same seed should return the same result. Using a different seed can result in different output.
seed = 7

# Using train_test_split to split data into X training/validation, Y training/validation
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test Harness
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Build Models
models = []

# Linear Models
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))

# Non-Linear Models
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

