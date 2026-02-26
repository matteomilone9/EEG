"""
Caricamento EDF, filtraggio, creazione etichette
"""
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


# In _preprocessing.py, modifica load_eeg_data:

def load_eeg_data(edf_path):
    """Carica file EDF e restituisce DataFrame con valori in microvolt"""
    print(f"📂 Caricamento {edf_path}...")
    
    edf = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Estrai dati
    data = edf.get_data()
    ch_names = edf.ch_names
    
    print(f"   Dati grezzi - range: [{data.min():.6e}, {data.max():.6e}]")
    
    # CONVERSIONE: millivolt → microvolt (x1000)
    data = data * 1000  # <-- CAMBIA DA 1e6 A 1000!
    
    print(f"   Dopo conversione - range: [{data.min():.2f}, {data.max():.2f}] µV")
    
    # Crea DataFrame
    df = pd.DataFrame(data, index=ch_names)
    
    # Rimuovi canali non EEG
    for ch in ['CPz', '']:
        if ch in df.index:
            df = df.drop(ch)
            print(f"   - Rimosso canale {ch}")
    
    # Statistiche
    print(f"\n📊 Dati caricati:")
    print(f"   - Canali: {df.shape[0]}")
    print(f"   - Campioni: {df.shape[1]}")
    print(f"   - Range: [{df.values.min():.2f}, {df.values.max():.2f}] µV")
    print(f"   - Media: {df.values.mean():.2f} µV")
    print(f"   - Std: {df.values.std():.2f} µV")
    
    return df


def filter_eeg(df, lowcut=7, highcut=40, fs=500):
    """Applica filtro band-pass"""
    sos = butter(4, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    df_filt = df.copy()
    for i in df.index:
        df_filt.loc[i] = sosfilt(sos, df.loc[i])
    return df_filt


def create_labels(events_path, fs=500):
    """Crea etichette dal file events"""
    events = pd.read_csv(events_path, sep='\t')

    # Espandi ogni evento in campioni
    labels = []
    for i in range(len(events)):
        n_samples = int(events['duration'][i] / (1000 / fs))  # ms → samples
        if events['value'][i] == 2:  # immaginazione motoria
            labels.extend([1] * n_samples)
        else:  # riposo/istruzione
            labels.extend([0] * n_samples)

    return np.array(labels)


# In _preprocessing.py, crea una versione senza filtro per debug

def load_eeg_data_no_filter(edf_path):
    """Carica EEG senza applicare filtro (per debug)"""
    edf = mne.io.read_raw_edf(edf_path)
    df = pd.DataFrame(edf.get_data())
    df.index = edf.ch_names
    
    # Solo rimozione canali non EEG
    if 'CPz' in df.index:
        df = df.drop('CPz')
    if '' in df.index:
        df = df.drop('')
    
    # Normalizzazione semplice
    df = (df - df.mean(axis=1, keepdims=True)) / (df.std(axis=1, keepdims=True) + 1e-8)
    
    print(f"🔍 Dopo caricamento - Media: {df.values.mean():.6f}")
    print(f"🔍 Dopo caricamento - Std: {df.values.std():.6f}")
    
    return df