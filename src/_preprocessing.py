# preprocessing.py
"""
Caricamento EDF, filtraggio, creazione etichette
"""
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

# Canali da rimuovere: non EEG, artefattuali, senza range fisico
CHANNELS_TO_DROP = ['CPz', '', 'HEOL', 'HEOR']


def load_eeg_data(edf_path):
    """Carica file EDF e restituisce DataFrame con valori in microvolt"""
    print(f"📂 Caricamento {edf_path}...")

    edf  = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = edf.get_data()       # shape: (n_ch, n_samples) — in Volt
    ch_names = edf.ch_names
    n_ch, n_samples = data.shape

    print(f"   Dati grezzi    : range=[{data.min():.3e}, {data.max():.3e}]  std={data.std():.3e}")

    # Conversione Volt → µV
    data = data * 1e6

    print(f"   Dopo conv. µV  : range=[{data.min():.2f}, {data.max():.2f}]  std={data.std():.2f} µV")

    # Verifica ampiezza plausibile
    if data.std() < 1.0:
        print("   ⚠️  Std < 1 µV dopo conversione — verifica la scala del file EDF")
    elif data.std() > 500:
        print("   ⚠️  Std > 500 µV — possibile artefatto o scala errata")
    else:
        print("   ✅ Ampiezza segnale plausibile")

    # Crea DataFrame (canali × campioni)
    df = pd.DataFrame(data, index=ch_names)

    # Rimuovi canali non EEG / artefattuali
    for ch in CHANNELS_TO_DROP:
        if ch in df.index:
            df = df.drop(ch)
            print(f"   - Rimosso canale: '{ch}'")

    print(f"\n📊 Dati caricati:")
    print(f"   Canali   : {df.shape[0]}  → {list(df.index)}")
    print(f"   Campioni : {df.shape[1]}")
    print(f"   Range    : [{df.values.min():.2f}, {df.values.max():.2f}] µV")
    print(f"   Media    : {df.values.mean():.4f} µV")
    print(f"   Std      : {df.values.std():.2f} µV")

    return df


def filter_eeg(df, lowcut=7, highcut=40, fs=500):
    """Applica filtro band-pass Butterworth ordine 4"""
    print(f"🔧 Filtraggio band-pass [{lowcut}–{highcut}] Hz...")
    sos      = butter(4, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    df_filt  = df.copy()
    for ch in df.index:
        df_filt.loc[ch] = sosfilt(sos, df.loc[ch].values)

    print(f"   Dopo filtro — std={df_filt.values.std():.2f} µV")
    return df_filt


def create_labels(events_path, total_samples, fs=500):
    """
    Crea etichette campione-per-campione dal file events TSV.

    Label:
        0  → Riposo esplicito       (value=3)
        1  → Motor Imagery          (value=2)
        2  → Preparazione/cue       (value=1)  → da scartare nel Dataset
       -1  → Non coperto da eventi  → da scartare nel Dataset

    Parameters
    ----------
    events_path   : path al file .tsv
    total_samples : numero totale di campioni del segnale EEG
    fs            : frequenza di campionamento (Hz)
    """
    events = pd.read_csv(events_path, sep='\t')
    ms_per_sample = 1000 / fs

    labels = np.full(total_samples, -1, dtype=int)

    label_map = {
        2: 1,   # Motor Imagery    → 1
        3: 0,   # Riposo           → 0
        1: 2,   # Preparazione/cue → 2 (scartata nel Dataset)
    }

    for _, row in events.iterrows():
        onset_s = int(row['onset']    / ms_per_sample)
        dur_s   = int(row['duration'] / ms_per_sample)
        end_s   = min(onset_s + dur_s, total_samples)
        val     = int(row['value'])
        lbl     = label_map.get(val, -1)
        labels[onset_s:end_s] = lbl

    # Statistiche
    vals, counts = np.unique(labels, return_counts=True)
    label_names  = {-1: "Non assegnato", 0: "Riposo", 1: "Motor Imagery", 2: "Preparazione"}
    print(f"\n🏷️  Labels create ({total_samples} campioni totali):")
    for v, c in zip(vals, counts):
        name = label_names.get(int(v), f"label={v}")
        print(f"   {name:20s}: {c:6d} campioni ({100*c/total_samples:.1f}%)")

    return labels
