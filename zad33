import numpy as np
import matplotlib.pyplot as plt

# --- ZADANIE 1: Wczytanie danych z pliku CSV ---
# Używamy dtype=str, aby wczytać wszystko jako tekst i uniknąć błędów przy kolumnach z nazwami.
# Parametr encoding upewnia się, że polskie znaki z nazw działów zostaną poprawnie wczytane.
dane = np.loadtxt("transakcje.csv", delimiter=",", skiprows=1, dtype=str, encoding='utf-8')

# Zakładając standardową kolejność z Twojego pliku: 
# kolumna 3 (indeks 3) to nazwa oddziału, kolumna 4 (indeks 4) to kwota PLN
nazwy_dzialow = dane[:, 3]
kwoty_str = dane[:, 4]

# Wyciągamy pierwszą cyfrę znaczącą (różną od zera) dla każdej transakcji.
# np.zeros utworzy nam tablicę na wyniki.
pierwsze_cyfry = np.zeros(len(kwoty_str), dtype=int)

for i, kwota in enumerate(kwoty_str):
    # Przeszukujemy każdy znak w kwocie od lewej do prawej
    for znak in kwota:
        if znak in '123456789': # Ignoruje '0', '.', '-' i bierze pierwszą cyfrę 1-9
            pierwsze_cyfry[i] = int(znak)
            break

# Unikalne nazwy działów (posortowane automatycznie przez numpy)
unikalne_dzialy = np.unique(nazwy_dzialow)

# --- ZADANIE 2: Prawo Benforda ---
cyfry = np.arange(1, 10)
# Obliczamy prawdopodobieństwa ze wzoru
p_benford = np.log10(1 + 1 / cyfry)

# --- ZADANIE 4: Wartość krytyczna ---
chi2_crit = 15.507

# Przygotowanie miejsca na 4 wykresy (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, dzial in enumerate(unikalne_dzialy):
    # Tworzymy maskę logiczną, żeby wybrać tylko cyfry dla bieżącego działu
    maska_dzialu = (nazwy_dzialow == dzial)
    cyfry_dzial = pierwsze_cyfry[maska_dzialu]
    N = len(cyfry_dzial) # Łączna liczba transakcji w dziale
    
    # 1. Zliczenie wystąpień poszczególnych cyfr (od 1 do 9) 
    # Funkcja np.bincount liczy wystąpienia (0, 1, 2... 9), więc bierzemy wycinek [1:10]
    obserwacje = np.bincount(cyfry_dzial, minlength=10)[1:10]
    
    # 2. Obliczenie wartości oczekiwanych Ed
    oczekiwane = p_benford * N
    
    # --- ZADANIE 3: Obliczenie statystyki chi-kwadrat ---
    chi2 = np.sum((obserwacje - oczekiwane)**2 / oczekiwane)
    
    # Rysowanie wykresów
    ax = axes[i]
    ax.bar(cyfry, obserwacje, color='skyblue', edgecolor='black', label='Rozkład empiryczny ($O_d$)')
    ax.plot(cyfry, oczekiwane, color='red', marker='o', linestyle='-', linewidth=2, label='Prawo Benforda ($E_d$)')
    
    # --- ZADANIE 4 (Ciąg dalszy): Ocena na podstawie wartości krytycznej ---
    if chi2 > chi2_crit:
        kolor_tytulu = 'crimson'
        wniosek = "ODRZUCAMY $H_0$ (Anomalia!)"
        ax.set_facecolor('#fff0f0') # Podświetlenie podejrzanego działu
    else:
        kolor_tytulu = 'forestgreen'
        wniosek = "Zgodny z Benfordem"
        
    ax.set_title(f"Dział: {dzial}\n$\chi^2 = {chi2:.3f}$ | {wniosek}", color=kolor_tytulu, fontsize=12, fontweight='bold')
    ax.set_xlabel('Pierwsza cyfra znacząca')
    ax.set_ylabel('Liczba transakcji')
    ax.set_xticks(cyfry)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
