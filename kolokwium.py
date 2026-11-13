import numpy as np
import matplotlib.pyplot as plt

dane = np.loadtxt("transakcje.csv", delimiter = ",", skiprows=1, dtype=str)

id = dane[:,0]
data = dane[:,1]
oddzial = dane[:,2]
nazwa_oddzialu = dane[:,3]
kwota_PLN = dane[:,4]

unikalne_dzialy = np.unique(nazwa_oddzialu)

pierwsze_cyfry = np.zeros(len(kwota_PLN), dtype=int)

for i, kwota in enumerate(kwota_PLN):
    for znak in kwota:
        if znak in '123456789':
            pierwsze_cyfry[i] = int(znak)
            break

# print(pierwsze_cyfry)

### Benford

cyfry = np.arange(1, 10)
benford = np.log10(1+1/cyfry)

### statystyka testowa

krytyczna = 15.507


###################

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, dzial in enumerate(unikalne_dzialy):
    maska = (nazwa_oddzialu == dzial)
    cyfry_dzial = pierwsze_cyfry[maska]
    N = len(cyfry_dzial)

    obserwacje = np.bincount(cyfry_dzial, minlength=10)[1:10]
    oczekiwane = benford * N

    chi2 = np.sum((obserwacje - oczekiwane)**2 / oczekiwane)
    
    ax = axes[i]
    ax.bar(cyfry, obserwacje, color='skyblue', edgecolor='black', label='Rozkład empiryczny')
    ax.plot(cyfry, oczekiwane, color='red', marker='o', linestyle='-', linewidth=2, label='Prawo Benforda')

    if chi2 > krytyczna:
        kolor_tytulu = 'crimson'
        wniosek = "Odrzucamy"
        ax.set_facecolor('#fff0f0')

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