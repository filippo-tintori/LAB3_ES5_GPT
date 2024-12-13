import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import soundfile # NON SERVE

# link utili - URL GLOBALI

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

# funz

def apriAudio(nome_file):
    """Apre un file audio e restituisce la frequenza di campionamento e i dati."""
    if nome_file.endswith(".wav"):
        freq_camp, dati = wav.read(nome_file)
    elif nome_file.endswith(".txt"):
        dati = np.loadtxt(nome_file)
        freq_camp = 44100 # freq campionamento dato dal prof, non ho usato random
    else:
        raise ValueError("Formato di file non supportato. Usa .wav o .txt.")
    return freq_camp, dati

def plottaWaveform(freq_camp, dati):
    """Plotta la waveform di un file audio."""
    tempo = np.linspace(0, len(dati) / freq_camp, num=len(dati))
    plt.plot(tempo, dati)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.title("Waveform")
    plt.show()

def fftSegnale(dati):
    """Calcola la FFT del segnale."""
    fft_coeff = np.fft.fft(dati)
    potenza = np.abs(fft_coeff)**2
    return fft_coeff, potenza

def plottaFFT(fft_coeff, potenza):
    """Plotta potenza, parte reale e parte immaginaria dei coefficienti FFT."""
    freq = np.fft.fftfreq(len(fft_coeff))
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(freq, potenza)
    plt.title("Potenza")
    plt.xlabel("Frequenza")
    plt.ylabel("Potenza")

    plt.subplot(3, 1, 2)
    plt.plot(freq, np.real(fft_coeff))
    plt.title("Parte Reale")
    plt.xlabel("Frequenza")
    plt.ylabel("Ampiezza")

    plt.subplot(3, 1, 3)
    plt.plot(freq, np.imag(fft_coeff))
    plt.title("Parte Immaginaria")
    plt.xlabel("Frequenza")
    plt.ylabel("Ampiezza")

    plt.tight_layout()
    plt.show()

def risintetizzaSegnale(fft_coeff):
    """Ri-sintetizza il segnale usando la FFT inversa."""
    return np.fft.ifft(fft_coeff).real

def risintetizzaSeniCoseni(fft_coeff):
    """Ri-sintetizza il segnale usando seni e coseni."""
    n = len(fft_coeff)
    t = np.arange(n)
    segnale = np.zeros(n)
    for k in range(n):
        ampiezza = np.abs(fft_coeff[k])
        fase = np.angle(fft_coeff[k])
        segnale += ampiezza * np.cos(2 * np.pi * k * t / n + fase)
    return segnale

def mascheraRumore(fft_coeff, potenza):
    """Maschera il rumore ponendo a zero i coefficienti con potenza minima."""
    indice_min_potenza = np.argmin(potenza)
    fft_coeff[indice_min_potenza] = 0
    return fft_coeff

# def funz di main

def esercitazioneA():
    print("Esercitazione A: ") # da fare

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