#From Scikit import image data of handwritten numbers
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
#Items in the dataset digits
dir(digits)
#Image of the first five numbers
plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i])
#Train the data using Logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)
model.fit(X_train,y_train)
#Measure the accuracy of the model
model.score(X_test, y_test)
#Predict the any five image numbers
model.predict(digits.data[35:40])
for i in range(35,40):
    plt.matshow(digits.images[i])
#Checking the confusion matrix
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot = True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
