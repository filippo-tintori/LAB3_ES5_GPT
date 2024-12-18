import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import find_peaks
from tqdm import tqdm
import soundcard as sd
#import soundfile # NON SERVE
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import urllib.request
import sys, os
import tempfile

# imposto parametri di stampa 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 12,  # font
    "axes.titlesize": 14,  # titolo degli assi
    "axes.labelsize": 12,  # etichette degli assi
    "xtick.labelsize": 10,  # etichette degli assi x
    "ytick.labelsize": 10,  # etichette degli assi y
})

##############################
#  LINK UTILI - URL GLOBALI  #
##############################
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

##############################
#           AUDIO            #
##############################

def apriAudio(nome_file):
    """Apre un file audio (.wav) o dati (.txt) e restituisce la frequenza di campionamento e i dati."""
    print(nome_file)
    nome_file = str(nome_file)
    
    # Gestione dei file scaricati da URL
    if nome_file.startswith("http://") or nome_file.startswith("https://"):
        print("Il file è un URL, lo scarico...")
        temp_dir = tempfile.gettempdir()  
        temp_file_path = os.path.join(temp_dir, os.path.basename(nome_file)) # temporaneo
        urllib.request.urlretrieve(nome_file, temp_file_path)  
        nome_file = temp_file_path 
    
    # Gestione dei file .wav
    if nome_file.endswith(".wav"):
        freq_camp, dati = wav.read(nome_file)
    # Gestione dei file .txt
    elif nome_file.endswith(".txt"):
        dati = np.loadtxt(nome_file)
        freq_camp = 44100  # Frequenza di campionamento predefinita
    # Formato non supportato
    else:
        raise ValueError("Formato di file non supportato. Usa .wav o .txt.")
    
    print("File aperto e utilizzabile.")
    return freq_camp, dati

def ascoltaAudio(nome_file):
    """Apre e riproduce un file audio utilizzando soundcard."""
    try:
        freq_camp, dati = apriAudio(nome_file)
        print(f"Riproduzione del file '{nome_file}' con frequenza di campionamento {freq_camp} Hz.")
        
        # Seleziona la scheda audio predefinita per la riproduzione
        scheda_audio = sc.default_output_device()
        
        # Riproduce l'audio
        scheda_audio.play(dati, samplerate=freq_camp)
        print("Riproduzione completata.")
    except Exception as e:
        print(f"Errore durante la riproduzione dell'audio: {e}")
        
##############################
#           GRAFICI          #
##############################

def GraficoDiffOriginaleFiltrato(coefficienti_originali, coefficienti_filtrati, frequenza_campionamento): # NON PROVATA - VERIFICARE VALIDITÀ
    """
    Visualizza le frequenze mantenute (verdi) e rimosse (rosse) dopo il filtraggio.

    Parametri:
    - coefficienti_originali: Array con i coefficienti di Fourier originali.
    - coefficienti_filtrati: Array con i coefficienti di Fourier filtrati.
    - frequenza_campionamento: Frequenza di campionamento del segnale.
    """
    n = len(coefficienti_originali)
    frequenze = np.fft.fftfreq(n, d=1/frequenza_campionamento)

    potenza_originale = np.abs(coefficienti_originali)**2
    potenza_filtrata = np.abs(coefficienti_filtrati)**2

    mantenute = np.where(potenza_filtrata > 0)[0]
    rimosse = np.where(potenza_filtrata == 0)[0]

    plt.figure(figsize=(10, 6))
    plt.title("Frequenze usate")
    plt.scatter(frequenze[mantenute], potenza_originale[mantenute], color='green', label='Frequenze mantenute', s=10)
    plt.scatter(frequenze[rimosse], potenza_originale[rimosse], color='red', label='Frequenze di rumore', s=10)

    plt.xlabel("f (Hz)")
    plt.ylabel("Potenza (u.a.)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plottaWaveform(dati):
    """Plotta la waveform di un file audio."""
    plt.plot(dati[:, 0], dati[:, 1])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza (u.a.)")
    plt.title("Waveform")
    plt.show()
    
def plottaWAV(canale):
    freqcamp = 44100
    durata = len(canale) / freqcamp  # Durata in secondi
    tempo = np.linspace(0, durata, len(canale))
    plt.figure(figsize=(10, 5))
    plt.plot(tempo, canale, label="Forma d'onda", color = 'coral')
    plt.title("Forma d'onda.")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza (u. a.)")
    plt.legend()
    plt.show()

def IndiceAfreq(potenza, indice):
    f= np.fft.fftfreq(len(potenza),d=1/44100)
    freqi=f[indice]
    return freqi

def zoomPicchi(potenza):
    picchiTrovati, _ = find_peaks(potenza, height=1e14)
    picchiTrovati=picchiTrovati[:len(picchiTrovati)//2-1]

    num_picchi = len(picchiTrovati)
    cols = 3  # Numero di colonne
    rows = (num_picchi // cols) + (num_picchi % cols > 0)  # Calcola il numero di righe
    
    # Limita la dimensione della figura
    max_figsize = 10  # Limite massimo per la larghezza/altezza della figura
    figsize = (min(cols * 5, max_figsize), min(rows * 5, max_figsize))  # Imposta una dimensione più piccola
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    
    # Se ax è una matrice di più righe e colonne, lo appiattiamo in un array
    ax = ax.flatten()
    
    for i, picco in tqdm(enumerate(picchiTrovati)):
        start = max(0, picco - 50)  # Imposta il margine sinistro
        end = min(len(potenza), picco + 50)  # Imposta il margine destro
        # Traccia il grafico zoomato per ogni picco
        ax[i].plot(np.arange(start, end), potenza[start:end])
        ax[i].set_title(f"Zoom Picco {i + 1} (Posizione {picco})")
        ax[i].set_xlabel("Indice")
        ax[i].set_ylabel("Potenza")
        ax[i].grid()

    # Rimuove gli assi vuoti, se necessario
    for j in range(num_picchi, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()

def zoomPicchiFrequenza(potenza, frequenza_campionamento=44100, zoom_range=50):
    """
    Zoom sui picchi della potenza con i grafici dei primi picchi messi uno accanto all'altro.
    
    Args:
        potenza (array): Spettro di potenza.
        frequenza_campionamento (int): Frequenza di campionamento (Hz). Default 44100 Hz.
        zoom_range (int): Numero di punti da considerare attorno al picco per lo zoom.
    """
    # Calcolo le frequenze
    frequenze = np.fft.fftfreq(len(potenza), d=1/frequenza_campionamento)
    
    # Trovo i picchi
    peaks, _ = find_peaks(potenza, height=1e14)  # Soglia minima per i picchi
    freq_peaks = frequenze[peaks]
    potenza_peaks = potenza[peaks]

    # Considera solo la prima metà dei picchi
    num_picchi = len(peaks)
    if num_picchi == 0:
        print("Nessun picco rilevato!")
        return

    metà_picchi = (num_picchi + 1) // 2 # tolgo i coeff negativi
    peaks = peaks[:metà_picchi]
    freq_peaks = freq_peaks[:metà_picchi]
    potenza_peaks = potenza_peaks[:metà_picchi]

    # Creazione del grafico
    fig, axs = plt.subplots(1, metà_picchi, figsize=(5 * metà_picchi, 5), constrained_layout=True)

    if metà_picchi == 1:
        axs = [axs]

    for i, peak_idx in enumerate(peaks):
        # Definizione dei limiti dello zoom
        start = max(0, peak_idx - zoom_range)
        end = min(len(potenza), peak_idx + zoom_range)
        
        # Plot per ogni picco
        axs[i].plot(frequenze[start:end], potenza[start:end], label="Zoom", color="royalblue")
        axs[i].scatter([freq_peaks[i]], [potenza_peaks[i]], color='red', label='Picco')
        axs[i].set_title(f"Picco a {freq_peaks[i]:.2f} Hz")
        axs[i].set_xlabel("Frequenza (Hz)")
        axs[i].set_ylabel("Potenza")
        axs[i].legend()
        axs[i].grid(True)

    # Mostra il grafico
    plt.suptitle("Zoom sui Picchi della Potenza (Prima metà)")
    plt.show()



"""def zoomPicchiFrequenza(potenza):
    picchiTrovati, _ = find_peaks(potenza, height=1e14)
    picchiTrovati = picchiTrovati[:len(picchiTrovati)//2-1]
    pot= potenza[picchiTrovati]
    picchiTrovati = [IndiceAfreq(potenza, picchi) for picchi in picchiTrovati] # conversione

    num_picchi = len(picchiTrovati)
    cols = 3  # Numero di colonne
    rows = (num_picchi // cols) + (num_picchi % cols > 0)  # Calcola il numero di righe
    
    # Limita la dimensione della figura
    max_figsize = 10  # Limite massimo per la larghezza/altezza della figura
    figsize = (min(cols * 5, max_figsize), min(rows * 5, max_figsize))  # Imposta una dimensione più piccola
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    
    # Se ax è una matrice di più righe e colonne, lo appiattiamo in un array
    ax = ax.flatten()
    
    for i, picco in tqdm(enumerate(picchiTrovati)):
        # Traccia il grafico zoomato per ogni picco
        ax[i].plot(picchiTrovati[i], pot[i])
        ax[i].set_title(f"Zoom Picco {i + 1} (Posizione {np.round(picco, 2)} (Hz))")
        ax[i].set_xlabel("Frequenza (Hz)")
        ax[i].set_ylabel("Potenza (u.a.)")
        ax[i].grid()

    # Rimuove gli assi vuoti, se necessario
    for j in range(num_picchi, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()"""


def salvaCanale(dati, frequenza_campionamento, file_output):
    wav.write(file_output, frequenza_campionamento, dati)
    
def fftSegnale(dati):
    """Calcola la FFT del segnale."""
    fft_coeff = np.fft.fft(dati[:,1])
    potenza = np.abs(fft_coeff) ** 2
    return fft_coeff, potenza 
   
def fftSegnaleB1(dati):
    """Calcola la FFT del segnale."""
    fft_coeff = np.fft.fft(dati)
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
    plt.ylabel("Potenza (u.a.)")

    plt.subplot(3, 1, 2)
    plt.plot(freq[:len(fft_coeff)//2], np.real(fft_coeff[:len(fft_coeff)//2]))
    plt.title("Parte Reale")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza (u.a.)")

    plt.subplot(3, 1, 3)
    plt.plot(freq[:len(fft_coeff)//2], np.imag(fft_coeff[:len(fft_coeff)//2]))
    plt.title("Parte Immaginaria")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza (u.a.)")

    plt.tight_layout()
    plt.show()


##############################
#      MASCHERA RUMORE       #
##############################
def mascheraRumore(fft_coeff, indice):
    """Rimuove i coefficienti che portano rumore."""
    potenza = np.abs(fft_coeff) ** 2
    #fft_coeff_filtrati = np.where(potenza > soglia, fft_coeff, 0)
    indiciPicchi, _ = find_peaks(potenza, height=1e8)
    picchi = potenza[indiciPicchi]
    print(f"Picchi trovati: {picchi}")
    fft_coeff_filtrati = np.zeros_like(fft_coeff) 

    # scelta per ogni file
    if indice == 1:
        piccoScelto = indiciPicchi[np.argmin(potenza[indiciPicchi])] # min = preservo il picco con potenza minore
        fft_coeff_filtrati[piccoScelto] = fft_coeff[piccoScelto] # azzero altri 

    if indice == 2:
        #piccoScelto = picchi[12] # valore che voglio togliere
        picchi_scelti = indiciPicchi[:12]
        #fft_coeff_filtrati[~np.isin(np.abs(fft_coeff)**2, picchi_scelti)]=0 # azzero questo
        for index, picco in enumerate(picchi_scelti):
            fft_coeff_filtrati[picco] = fft_coeff[picco]
        
    if indice == 3:
         #piccoScelto = picchi[12] # valore che voglio togliere
        print(picchi)
        picchi_scelti = indiciPicchi[1:2]
        #fft_coeff_filtrati[~np.isin(np.abs(fft_coeff)**2, picchi_scelti)]=0 # azzero questo
        for index, picco in enumerate(picchi_scelti):
            fft_coeff_filtrati[picco] = fft_coeff[picco]
        
    return fft_coeff_filtrati

# migliorare il tempo di esecuzione del programma - per ora neglio ordini dei min. ___> integrare scipy



##############################
#       RISINTETIZZA         #
##############################

def risintetizzaSegnale(fft_coeff):
    """Ri-sintetizza il segnale usando la FFT inversa."""
    return np.fft.ifft(fft_coeff).real

def risintetizzaSeniCoseni(fft_coeff):
    """Ri-sintetizza il segnale usando seni e coseni."""
    t_index = len(fft_coeff)
    segnale = np.zeros(t_index)
    fft_coeff=fft_coeff[:len(fft_coeff)//2-1]

    #  DataFrame
    df = pd.DataFrame({
        'indice': np.arange(len(fft_coeff)),
        'coeff_reale': np.real(fft_coeff),
        'coeff_immaginario': np.imag(fft_coeff),
        'potenza': np.abs(fft_coeff) ** 2
    })
    df_filtrato = df[df['potenza'] > 0]
    
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



def plottaRisintonizzata(dati_originali, dati_filtrati, index):
    """Plotta il confronto tra segnale originale e filtrato con zoom su un'area."""
    tempo = dati_originali[:, 0]
    
    # Creazione della figura principale
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tempo, dati_originali[:, 1], label="Originale", color="dodgerblue")
    ax.plot(tempo, dati_filtrati, label="Filtrato", alpha=0.7, color="coral")
    
    # Etichette e titolo
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ampiezza (u.a.)")
    ax.set_title("Confronto tra segnale originale e filtrato")
    ax.legend(loc="upper left")
    
    # Aggiunta dello zoom
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=1)
    
    # Zoomare sull'intervallo x da 0.1 a 0.2
    axins.plot(tempo, dati_originali[:, 1], label="Originale")
    axins.plot(tempo, dati_filtrati, label="Filtrato", alpha=0.7)
    
    # Impostazioni dell'area zoomata
    if index == 3:
        axins.set_xlim(0.1, 0.11)
    else:
        axins.set_xlim(0.1, 0.2)
    
    axins.set_ylim(min(dati_originali[:, 1]), max(dati_originali[:, 1]))
    
    plt.show()



##############################
#      ESERCITAZIONE A       #
##############################

def esercitazioneA(parte):
    index = int(parte)
    file = [index-1]
    print(f"Elaborazione del file: {file}")
    
    freq_camp, dati = apriAudio(file)
    plottaWaveform(dati)
    
    fft_coeff, potenza = fftSegnale(dati)
    plottaFFT(fft_coeff, potenza)
    
    fft_filtrato = mascheraRumore(fft_coeff, index)
    
    segnale_fft = risintetizzaSegnale(fft_filtrato)
    segnale_seni_coseni = risintetizzaSeniCoseni(fft_filtrato)

    plottaRisintonizzata(dati, segnale_fft, index=index) # ifft
    plottaRisintonizzata(dati, segnale_seni_coseni, index=index) #seni e coseni



##############################
#      ESERCITAZIONE B1      #
##############################

def esercitazioneB1(parte):
    index = int(parte)
    file = parteB[0][index-1]
    
    if parte == "1":
        freq_camp, dati = apriAudio(file)
        dati=dati[:,0]
        plottaWAV(dati)
        salvaCanale(dati, 44100, "/Users/filippo/Documenti/UniPG/3°Anno/Laboratorio di Elettronica e Tecniche di Acquisizione Dati/Relazione5/LAB3_ES5_GPT/copia.wav")
        coeff_fft, pot = fftSegnaleB1(dati)
        print(len)
        plottaFFT(coeff_fft, pot)
        zoomPicchi(pot)
        zoomPicchiFrequenza(pot)
        
    elif parte == "2":
        freq_camp, dati = apriAudio(file)
        fft_coeff, potenza = fftSegnale(dati)
        plottaFFT(fft_coeff, potenza)
        
    elif parte == "3":
        freq_camp, dati = apriAudio(file)
        fft_coeff, potenza = fftSegnale(dati)
        fft_coeff_filtrato = mascheraRumore(fft_coeff, potenza)
        segnale_filtrato = risintetizzaSegnale(fft_coeff_filtrato)
        print("Segnale filtrato sintetizzato usando FFT inversa.")
        
    elif parte == "4":
        freq_camp, dati = apriAudio(file)
        fft_coeff, potenza = fftSegnale(dati)
        fft_coeff_filtrato = mascheraRumore(fft_coeff, potenza)
        segnale_filtrato = risintetizzaSegnale(fft_coeff_filtrato)
        print("Segnale filtrato sintetizzato usando FFT inversa.")
        
    elif parte == "5":
        freq_camp, dati = apriAudio(file)
        fft_coeff, potenza = fftSegnale(dati)
        fft_coeff_filtrato = mascheraRumore(fft_coeff, potenza)
        segnale_filtrato = risintetizzaSegnale(fft_coeff_filtrato)
        print("Segnale filtrato sintetizzato usando FFT inversa.")
        
    else:
        print("Parte non riconosciuta.")
        
##############################
#      ESERCITAZIONE B2      #
##############################

def esercitazioneB2(parte):
    index = int(parte)
    file = parteB[1][index-1]
    
    if parte == "1":
        freq_camp, dati = apriAudio(file)
        plottaWav(dati)
        
    elif parte == "2":
        freq_camp, dati = apriAudio(file)
        fft_coeff, potenza = fftSegnale(dati)
        plottaFFT(fft_coeff, potenza)
        
    else:
        print("Parte non riconosciuta.")

##############################
#      ESERCITAZIONE B3      #
##############################

def esercitazioneB3(parte):
    index = int(parte)
    file = parteB[2][index-1]
    
    if parte == "1":
        freq_camp, dati = apriAudio(file)
        plottaWaveform(freq_camp, dati)
        
    elif parte == "2":
        freq_camp, dati = apriAudio(file)
        fft_coeff, potenza = fftSegnale(dati)
        plottaFFT(fft_coeff, potenza)

    else:
        print("Parte non riconosciuta.")
        
##############################
#             MAIN           #
##############################

def main():
    parser = argparse.ArgumentParser(description="Esercitazioni audio.")
    parser.add_argument("esercitazione", choices=["A", "B1", "B2", "B3"], help="Seleziona l'esercitazione.")
    parser.add_argument("parte", nargs="?", help="Seleziona la parte dell'esercitazione.")
    args = parser.parse_args()

    if args.esercitazione == "A":
        if args.parte:
            esercitazioneA(args.parte)
        else:
            print("Per l'esercitazione A, specificare una parte (1, 2 o 3).")

    elif args.esercitazione == "B1":
        if args.parte:
            esercitazioneB1(args.parte)
        else:
            print("Per l'esercitazione B, specificare una parte (da 1 a 5).")
    elif args.esercitazione == "B2":
        if args.parte:
            esercitazioneB2(args.parte)
        else:
            print("Per l'esercitazione B, specificare una parte (1 o 2).")
    elif args.esercitazione == "B3":
        if args.parte:
            esercitazioneB3(args.parte)
        else:
            print("Per l'esercitazione B, specificare una parte (1 o 2).")


if __name__ == "__main__":
    main()