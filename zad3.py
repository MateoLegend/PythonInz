import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- Tydzień 3: ZADANIE 1 ---
digits = load_digits()
X = digits.data
y = digits.target

# Przedstawienie przykładowego obrazu w postaci macierzy wartości pikseli
print("Przykładowa macierz pikseli (cyfra", y[0], "):")
print(X[0].reshape(8, 8))

# Wyświetlenie kilku przykładowych obrazów
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    axes[i].imshow(X[i].reshape(8, 8), cmap='gray_r')
    axes[i].set_title(f"Etykieta: {y[i]}")
    axes[i].axis('off')
plt.suptitle("Przykładowe obrazy ze zbioru danych")
plt.show()

# --- ZADANIE 2 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ZADANIE 3 i 4: Implementacja KNN ---
def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for x in X_test:
        # Obliczenie odległości euklidesowej od badanego obrazu 'x' do wszystkich 'X_train' ze wzoru d(x,y)
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        
        # Znalezienie indeksów k najbliższych sąsiadów
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        
        # Przypisanie najczęstszej etykiety
        values, counts = np.unique(nearest_labels, return_counts=True)
        y_pred.append(values[np.argmax(counts)])
    return np.array(y_pred)

# Testowanie dla k=3
y_pred_k3 = knn_predict(X_train, y_train, X_test, k=3)
accuracy_k3 = np.sum(y_pred_k3 == y_test) / len(y_test)
print(f"\nSkuteczność klasyfikacji KNN dla k=3: {accuracy_k3:.4f} (ok. {accuracy_k3*100:.1f}%)")

# --- ZADANIE 5: Skuteczność w zależności od rozmiaru ---
rozmiary_zbioru = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
wartosci_k = [1, 3, 7, 15]
wyniki_knn = {k: [] for k in wartosci_k}
rzeczywiste_rozmiary = []

for frac in rozmiary_zbioru:
    limit = int(frac * len(X_train))
    X_train_sub, y_train_sub = X_train[:limit], y_train[:limit]
    rzeczywiste_rozmiary.append(limit)
    
    for k in wartosci_k:
        y_pred_sub = knn_predict(X_train_sub, y_train_sub, X_test, k)
        acc = np.sum(y_pred_sub == y_test) / len(y_test)
        wyniki_knn[k].append(acc)

plt.figure(figsize=(10, 6))
for k in wartosci_k:
    plt.plot(rzeczywiste_rozmiary, wyniki_knn[k], marker='o', label=f'KNN (k={k})')
plt.xlabel("Liczba przykładów treningowych")
plt.ylabel("Skuteczność (Accuracy)")
plt.title("Zależność skuteczności od liczby przykładów i parametru k")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()