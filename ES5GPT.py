import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from tqdm import tqdm
#import soundfile # NON SERVE

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
    print("File aperto e utilizzabile.")
    return freq_camp, dati

def plottaWaveform(freq_camp, dati):
    """Plotta la waveform di un file audio."""
    tempo = np.linspace(0, len(dati) / freq_camp, num=len(dati))
    plt.plot(dati[:,0], dati[:,1])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.title("Waveform")
    plt.show()
    
def plottaRisintonizzata(dati, amp):
    """Plotta la waveform risintonizzata di un file audio."""
    plt.plot(dati[:,0], amp)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.title("Waveform della risintonizzata")
    plt.show()

def fftSegnale(dati):
    """Calcola la FFT del segnale."""
    fft_coeff = np.fft.fft(dati[:,1])
    potenza = np.abs(fft_coeff)**2
    return fft_coeff, potenza

def plottaFFT(fft_coeff, potenza):
    """Plotta potenza, parte reale e parte immaginaria dei coefficienti FFT."""
    freq = np.fft.fftfreq(len(fft_coeff))
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(freq[:len(fft_coeff)//2], potenza[:len(fft_coeff)//2])
    plt.title("Potenza")
    plt.xlabel("Frequenza")
    plt.ylabel("Potenza")   
    print(np.real(fft_coeff[:len(fft_coeff)//2]))

    plt.subplot(3, 1, 2)
    plt.plot(freq[:len(fft_coeff)//2], np.real(fft_coeff[:len(fft_coeff)//2]))
    plt.title("Parte Reale")
    plt.xlabel("Frequenza")
    plt.ylabel("Ampiezza")

    plt.subplot(3, 1, 3)
    plt.plot(freq[:len(fft_coeff)//2], np.imag(fft_coeff[:len(fft_coeff)//2]))
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
    # coefficienti con modulo maggiore di una soglia
    soglia = 1e-6
    print(f"FFT originale: {len(fft_coeff)} coefficienti")
    fft_coeff_filtrati = fft_coeff[np.abs(fft_coeff) > soglia]
    print(f"Numero di coefficienti non nulli: {len(fft_coeff_filtrati)}")

    # Ri-sintetizza il segnale
    t_index = len(fft_coeff)
    segnale = np.zeros(t_index)
    for t in tqdm(range(t_index)):
        somma = 0
        for k, coeff in enumerate(fft_coeff_filtrati):
            somma += (
                np.real(coeff) * np.cos(2 * np.pi * k * t / t_index)
                - np.imag(coeff) * np.sin(2 * np.pi * k * t / t_index)
            ) / t_index
        segnale[t] = somma
    return segnale

def mascheraRumore(fft_coeff, potenza, soglia=1e-6):
    """Maschera il rumore ponendo a zero i coefficienti con potenza minima."""
    fft_coeff_filtrati = fft_coeff.copy()
    fft_coeff_filtrati[np.abs(fft_coeff_filtrati) < soglia] = 0
    return fft_coeff_filtrati

# def funz di main

def esercitazioneA():
    print("Esercitazione A: ") # da fare
    file = parteA[0]
    a,b = apriAudio(file)
    plottaWaveform(a,b)
    c, d = fftSegnale(b)
    plottaFFT(c,d)
    #antr=risintetizzaSegnale(c)
    #sign=risintetizzaSeniCoseni(c)
    #plottaRisintonizzata(b,sign)
    #plottaRisintonizzata(b,antr)
    e = mascheraRumore (c, d)
    ants=risintetizzaSegnale(e)
    signs=risintetizzaSeniCoseni(e)
    plottaRisintonizzata(b,signs)
    plottaRisintonizzata(b,ants)
    
    

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
        
# SI DEVE METTERE UN ATTRIBUTO PER FORZA

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