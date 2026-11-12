import numpy as np
import matplotlib.pyplot as plt

dane = np.loadtxt("cars.csv", skiprows=1)

fig, ax = plt.subplots(figsize=(8,5))
plt.hist(dane, bins=23, density=False)
plt.title("Dane")
plt.axis([5,29,0,70])
plt.show()

from scipy.special import factorial

# --- ZADANIE 2: Statystyki opisowe ---
srednia = np.mean(dane)
wariancja = np.var(dane)
odchylenie = np.std(dane)

print("--- ZADANIE 2 ---")
print(f"Średnia: {srednia:.2f}")
print(f"Wariancja: {wariancja:.2f}")
print(f"Odchylenie standardowe: {odchylenie:.2f}")
# Wniosek do zadania 2: Zauważ, że średnia i wariancja są bardzo blisko siebie (ok. 16.1 vs 17.0). 
# Jest to charakterystyczna cecha rozkładu Poissona, w którym wariancja = średnia.

# --- ZADANIE 3: Rozkład Poissona ---
# Parametr lambda ze wzoru zastępujemy policzoną średnią
lam = srednia

# Aby wyliczyć model i go porównać z danymi, wygodnie będzie wyciągnąć unikalne 
# wartości występujące w naszych danych (k) i policzyć ich częstotliwość występowania.
k_vals, obserwacje = np.unique(dane, return_counts=True)

# Obliczamy prawdopodobieństwo P(X=k) ze wzoru podanego w PDF
prawdopodobienstwa = (lam**k_vals * np.exp(-lam)) / factorial(k_vals)

# --- ZADANIE 4: Porównanie rozkładów ---
N = len(dane)  # całkowita liczba obserwacji
oczekiwane_obserwacje = N * prawdopodobienstwa

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Histogram empiryczny, ale tym razem używamy wyciągniętych konkretnych zliczeń (słupki)
ax2.bar(k_vals, obserwacje, alpha=0.6, color='skyblue', edgecolor='black', label='Liczności z obserwacji')

# Model Poissona pokazany jako połączone czerwone kropki
ax2.plot(k_vals, oczekiwane_obserwacje, 'ro-', lw=2, label='Model Poissona (Oczekiwane)')

ax2.set_xlabel("Liczba samochodów na minutę (k)")
ax2.set_ylabel("Liczba obserwacji")
ax2.set_title("Porównanie obserwacji empirycznych z modelem rozkładu Poissona")
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
plt.show()

# --- ZADANIE 5: Prawdopodobieństwo (X >= 15) ---
# Szansa na 15 lub więcej = 1 - Szansa na(od 0 do 14 włącznie)
k_do_14 = np.arange(0, 15)
prob_do_14 = (lam**k_do_14 * np.exp(-lam)) / factorial(k_do_14)
prob_15_plus = 1 - np.sum(prob_do_14)

print("\n--- ZADANIE 5 ---")
print(f"Prawdopodobieństwo przejazdu >= 15 aut wynosi: {prob_15_plus:.4f} (ok. {prob_15_plus*100:.1f}%)")

# --- ZADANIE 6: Zależność od czasu ---
# Dzielimy płaską tablicę 720 obserwacji na dwuwymiarową: 12 godzin x 60 minut
godziny = dane.reshape(-1, 60)

# Liczymy średnie na wierszach (axis=1) - co daje 1 średnią na każdą godzinę
srednie_godzinowe = godziny.mean(axis=1)

# Wyliczamy względną zmianę w procentach 
zmiana_wzgledna = (srednie_godzinowe - lam) / lam * 100

numery_godzin = np.arange(1, len(srednie_godzinowe) + 1)

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.bar(numery_godzin, zmiana_wzgledna, color='coral', edgecolor='black')
ax3.axhline(0, color='black', lw=1) # Linia 0% reprezentująca średnią globalną (lam)

ax3.set_xlabel("Kolejna godzina obserwacji")
ax3.set_ylabel("Względna zmiana średniej [%]")
ax3.set_title("Zmiana intensywności ruchu w kolejnych godzinach (odchylenie od średniej globalnej)")
ax3.set_xticks(numery_godzin)
ax3.grid(axis='y', alpha=0.3)
plt.show()