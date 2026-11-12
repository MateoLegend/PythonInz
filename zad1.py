import numpy as np
import matplotlib.pyplot as plt

dane = np.loadtxt("sales.csv", delimiter = ",", skiprows=1, dtype=str)

daty = dane[:,0]
smartfony = dane[:,1].astype(float)
sluchawki = dane[:,2].astype(float)
tablety = dane[:,3].astype(float)
laptopy = dane[:,4].astype(float)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(daty, smartfony, label="smartfony")
ax.plot(daty, sluchawki, label="sluchawki")
ax.plot(daty, tablety, label="tablety")
ax.plot(daty, laptopy, label="laptopy")

ax.set_xlabel("Data")
ax.set_ylabel("Liczba sprzedanych egzemplarzy")
ax.set_title("Dzienna sprzedaż")
ax.grid(lw=0.25)
ax.legend(frameon=False)

ax.set_xticks(np.arange(0, len(daty), 30))
ax.set_xticklabels(daty[::30], rotation=45)

plt.tight_layout()
plt.show()

# zad2
okno = 7
wagi = np.ones(okno) / okno

srednia_smartfony = np.convolve(smartfony, wagi, mode='valid')
srednia_sluchawki = np.convolve(sluchawki, wagi, mode='valid')
srednia_tablety = np.convolve(tablety, wagi, mode='valid')
srednia_laptopy = np.convolve(laptopy, wagi, mode='valid')

daty_srednie = daty[okno-1:]

fig2, ax2 = plt.subplots(figsize=(10,6))

ax2.plot(daty, smartfony, alpha=0.3, color="blue")
ax2.plot(daty, sluchawki, alpha=0.3, color="orange")
ax2.plot(daty, tablety, alpha=0.3, color="green")
ax2.plot(daty, laptopy, alpha=0.3, color="red")

ax2.plot(daty_srednie, srednia_smartfony, label="Śr. 7 dni (Smartfony)", color="blue", lw=2)
ax2.plot(daty_srednie, srednia_sluchawki, label="Śr. 7 dni (Słuchawki)", color="orange", lw=2)
ax2.plot(daty_srednie, srednia_tablety, label="Śr. 7 dni (Tablety)", color="green", lw=2)
ax2.plot(daty_srednie, srednia_laptopy, label="Śr. 7 dni (Laptopy)", color="red", lw=2)

ax2.set_xlabel("Data")
ax2.set_ylabel("Liczba sprzedanych egzemplarzy")
ax2.set_title("Sprzedaż produktów z 7-dniową średnią kroczącą")
ax2.grid(lw=0.25)
ax2.legend(frameon=False)


ax2.set_xticks(np.arange(0, len(daty), 30))
ax2.set_xticklabels(daty[::30], rotation=45)
plt.tight_layout()
plt.show()

# zad3

sprzedaz_dzienna = smartfony + sluchawki + tablety + laptopy
liczba_tygodni = len(sprzedaz_dzienna) // 7
sprzedaz_pelne_tygodnie = sprzedaz_dzienna[:liczba_tygodni * 7]
sprzedaz_tygodniowa = sprzedaz_pelne_tygodnie.reshape(-1, 7).sum(axis=1)

numery_tygodni = np.arange(1, liczba_tygodni + 1)

tydzien_max = np.argmax(sprzedaz_tygodniowa) + 1
tydzien_min = np.argmin(sprzedaz_tygodniowa) + 1
wartosc_max = np.max(sprzedaz_tygodniowa)
wartosc_min = np.min(sprzedaz_tygodniowa)

print(f"Największą sprzedaż ({wartosc_max:.0f} szt.) odnotowano w {tydzien_max}. tygodniu.")
print(f"Najmniejszą sprzedaż ({wartosc_min:.0f} szt.) odnotowano w {tydzien_min}. tygodniu.")


fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.bar(numery_tygodni, sprzedaz_tygodniowa, color="teal", alpha=0.8)

ax3.bar(tydzien_max, wartosc_max, color="crimson", label=f"Najlepszy ({tydzien_max}. tydz.)")
ax3.bar(tydzien_min, wartosc_min, color="gold", label=f"Najgorszy ({tydzien_min}. tydz.)")

ax3.set_xlabel("Numer tygodnia")
ax3.set_ylabel("Łączna liczba sprzedanych produktów")
ax3.set_title("Całkowita sprzedaż produktów w kolejnych tygodniach roku")
ax3.set_xlim(0, liczba_tygodni + 1)
ax3.grid(axis='y', lw=0.25)
ax3.legend(frameon=False)

plt.tight_layout()
plt.show()

# zad4

import datetime

daty_dt = daty.astype('datetime64[D]')
dni_tygodnia_liczbowo = np.array([d.item().weekday() for d in daty_dt])
sprzedaz_w_dniach = np.zeros(7)

for i in range(7):
    maska_dnia = (dni_tygodnia_liczbowo == i)
    sprzedaz_w_dniach[i] = np.sum(sprzedaz_dzienna[maska_dnia])

nazwy_dni = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]

najlepszy_dzien_idx = np.argmax(sprzedaz_w_dniach)
najgorszy_dzien_idx = np.argmin(sprzedaz_w_dniach)

print(f"Najwięcej produktów sprzedano w: {nazwy_dni[najlepszy_dzien_idx]} ({sprzedaz_w_dniach[najlepszy_dzien_idx]:.0f} szt.)")
print(f"Najmniej produktów sprzedano w: {nazwy_dni[najgorszy_dzien_idx]} ({sprzedaz_w_dniach[najgorszy_dzien_idx]:.0f} szt.)")

calkowita_sprzedaz = np.sum(sprzedaz_w_dniach)
udzialy_procentowe = (sprzedaz_w_dniach / calkowita_sprzedaz) * 100

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.bar(nazwy_dni, udzialy_procentowe, color="mediumpurple", edgecolor="black", zorder=3)
ax4.set_xlabel("Dzień tygodnia")
ax4.set_ylabel("Udział w całkowitej sprzedaży [%]")
ax4.set_title("Procentowy udział poszczególnych dni tygodnia w całkowitej sprzedaży")
ax4.grid(axis='y', lw=0.25, zorder=0, alpha=0.7)
ax4.set_ylim(0, max(udzialy_procentowe) * 1.15)

plt.tight_layout()
plt.show()