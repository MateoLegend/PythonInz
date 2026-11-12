import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

digits = load_digits()

X = digits.data

y = digits.target

print("X:", X)
print("Y:", y)
print("Rozmiar zbioru danych:", X.shape)
print("Rozmiar etykiet:", y.shape)
print("Liczba przykładów:", len(X))
print("Liczba cech:", len(y))
print("Liczba klas:", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Liczba przykładów treningowych:", len(X_train))
print("Liczba przykładów testowych:", len(X_test))
print("Liczba Y treningowych:", len(y_train))
print("Liczba Y testowych:", len(y_test))

clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(len(y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()