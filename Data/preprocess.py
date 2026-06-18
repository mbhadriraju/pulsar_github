import numpy as np
import psrchive
import subprocess
import os

def load_data(path):
    arch = psrchive.Archive.load(path)

    arch.dedisperse()
    arch.remove_baseline()

    X_data = arch.get_data()[0, 0]

    return X_data, arch


def replace_data(path, new_data):
    arch = load_data(path)[1]

    subint = arch.get_Integration(0)
    nchan = subint.get_nchan()

    for ichan in range(nchan):
        prof = subint.get_Profile(0, ichan)
        amps = prof.get_amps()
        amps[:] = new_data[ichan, :]
    
    arch.unload(path)


def preprocess_individual(path, time_channels=1024, freq_channels=32, norm=True, rem_polar=True, roll=None, output_path=None):
    subprocess.run([
        "pam", "--setnbin", f"{time_channels}", "-m", f"{path}",
    ])

    subprocess.run([
        "pam", "--setnchn", f"{freq_channels}", "-m", f"{path}"
    ])

    if rem_polar:
        subprocess.run([
            "pam", "-pm", f"{path}"
        ])

    X_data, arch = load_data(path)

    mu_x = np.mean(X_data)
    sigma_x = np.std(X_data)

    X_data = (X_data - mu_x) / sigma_x if norm and sigma_x != 0 else X_data

    replace_data(path, X_data)

    return X_data

def preprocess_all(root_path, time_channels=1024, freq_channels=32, norm=True, rem_polar=True, roll=None, output_path=None):
    for dirname, _, filenames in os.walk(root_path):
        for name in filenames:
            if name.endswith('.txt'): continue
            file_path = os.path.join(dirname, name)
            print(f"Preprocessing {file_path}...")
            preprocess_individual(file_path, time_channels, freq_channels, norm, rem_polar, roll, output_path)