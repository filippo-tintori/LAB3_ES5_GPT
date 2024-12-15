import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Link utili - URL GLOBALI
parteA = [
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/data1.txt",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/data2.txt",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/data3.txt"
]

parteB = [
    ["https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/diapason.wav",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/pulita_semplice.wav",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/pulita_media.wav",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/pulita_difficile.wav",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/distorta.wav"],
    
    ["https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/pulita_pezzo.wav",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/distorta_pezzo.wav"],
    
    ["https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/primo.wav",
    "https://www.fisgeo.unipg.it/~duranti/laboratoriodue/laboratorio_24-25/_slides/secondo.wav"]
]

def apriAudio(nome_file):
    """Apre un file di testo e restituisce la frequenza di campionamento e i dati."""
    dati = np.loadtxt(nome_file)
    freq_camp = 44100  # Frequenza di campionamento fornita
    return freq_camp, dati

def plottaWaveform(dati):
    """Plotta la waveform di un file audio."""
    plt.plot(dati[:, 0], dati[:, 1])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.title("Waveform")
    plt.show()

def fftSegnale(dati):
    """Calcola la FFT del segnale."""
    fft_coeff = np.fft.fft(dati[:, 1])
    potenza = np.abs(fft_coeff) ** 2
    return fft_coeff, potenza

def plottaFFT(fft_coeff, potenza):
    """Plotta potenza, parte reale e parte immaginaria dei coefficienti FFT."""
    freq = np.fft.fftfreq(len(fft_coeff), d=1/44100)
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(freq[:len(fft_coeff)//2], potenza[:len(fft_coeff)//2])
    plt.title("Potenza")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Potenza")

    plt.subplot(3, 1, 2)
    plt.plot(freq[:len(fft_coeff)//2], np.real(fft_coeff[:len(fft_coeff)//2]))
    plt.title("Parte Reale")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza")

    plt.subplot(3, 1, 3)
    plt.plot(freq[:len(fft_coeff)//2], np.imag(fft_coeff[:len(fft_coeff)//2]))
    plt.title("Parte Immaginaria")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza")

    plt.tight_layout()
    plt.show()

def mascheraRumore(fft_coeff, soglia):
    """Rimuove i coefficienti con potenza sotto una soglia."""
    potenza = np.abs(fft_coeff) ** 2
    fft_coeff_filtrati = np.where(potenza > soglia, fft_coeff, 0)
    return fft_coeff_filtrati

def risintetizzaSegnale(fft_coeff):
    """Ri-sintetizza il segnale usando la FFT inversa."""
    return np.fft.ifft(fft_coeff).real

def risintetizzaSeniCoseni(fft_coeff):
    """Ri-sintetizza il segnale usando seni e coseni."""
    t_index = len(fft_coeff)
    segnale = np.zeros(t_index)

    #  DataFrame bello
    df = pd.DataFrame({
        'indice': np.arange(len(fft_coeff)),
        'coeff_reale': np.real(fft_coeff),
        'coeff_immaginario': np.imag(fft_coeff),
        'potenza': np.abs(fft_coeff) ** 2
    })
    df_filtrato = df[df['potenza'] > 0]  # no coeff nulli

    for t in tqdm(range(t_index)):
        somma = 0
        for _, row in df_filtrato.iterrows():
            k = row['indice']
            coeff_reale = row['coeff_reale']
            coeff_immaginario = row['coeff_immaginario']
            somma += (
                coeff_reale * np.cos(2 * np.pi * k * t / t_index)
                - coeff_immaginario * np.sin(2 * np.pi * k * t / t_index)
            )
        segnale[t] = somma / t_index
    return segnale

def plottaRisintonizzata(dati_originali, dati_filtrati):
    """Plotta il confronto tra segnale originale e filtrato."""
    tempo = dati_originali[:, 0]
    plt.plot(tempo, dati_originali[:, 1], label="Originale")
    plt.plot(tempo, dati_filtrati, label="Filtrato", alpha=0.7)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.title("Confronto tra segnale originale e filtrato")
    plt.legend()
    plt.show()

def esercitazioneA():
    for file in parteA:
        print(f"Elaborazione del file: {file}")
        freq_camp, dati = apriAudio(file)
        plottaWaveform(dati)
        
        fft_coeff, potenza = fftSegnale(dati)
        plottaFFT(fft_coeff, potenza)

        soglia = np.mean(potenza)
        print(soglia)
        fft_filtrato = mascheraRumore(fft_coeff, soglia)

        segnale_fft = risintetizzaSegnale(fft_filtrato)
        segnale_seni_coseni = risintetizzaSeniCoseni(fft_filtrato)

        plottaRisintonizzata(dati, segnale_fft)
        plottaRisintonizzata(dati, segnale_seni_coseni)
        
def esercitazioneB(parte):
    if parte == "1":
        freq_camp, dati = apriAudio("audio.wav")
        plottaWaveform(freq_camp, dati)
    elif parte == "2":
        freq_camp, dati = apriAudio("audio.wav")
        fft_coeff, potenza = fftSegnale(dati)
        plottaFFT(fft_coeff, potenza)
    elif parte == "3":
        freq_camp, dati = apriAudio("audio.wav")
        fft_coeff, potenza = fftSegnale(dati)
        fft_coeff_filtrato = mascheraRumore(fft_coeff, potenza)
        segnale_filtrato = risintetizzaSegnale(fft_coeff_filtrato)
        print("Segnale filtrato sintetizzato usando FFT inversa.")
    else:
        print("Parte non riconosciuta.")

def main():
    parser = argparse.ArgumentParser(description="Esercitazioni audio.")
    parser.add_argument("esercitazione", choices=["A", "B"], help="Seleziona l'esercitazione.")
    parser.add_argument("parte", nargs="?", help="Seleziona la parte dell'esercitazione (solo per B).")
    args = parser.parse_args()

    if args.esercitazione == "A":
        esercitazioneA()
    elif args.esercitazione == "B":
        if args.parte:
            esercitazioneB(args.parte)
        else:
            print("Per l'esercitazione B, specificare una parte (1, 2 o 3).")

if __name__ == "__main__":
    main()