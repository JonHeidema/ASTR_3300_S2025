import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from astropy.io import fits
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
import random
import os

def extract_window(wavelength, flux, center, window_size=50, num_points=100):
    mask = (wavelength >= center - window_size) & (wavelength <= center + window_size)
    
    if np.sum(mask) < 10:  # Not enough points, skip
        return None
    
    extracted_wavelength = wavelength[mask]
    extracted_flux = flux[mask]

    # Normalize extracted flux
    extracted_flux = (extracted_flux - np.mean(extracted_flux)) / np.std(extracted_flux)

    # Resample to fixed number of points
    f_interp = interp1d(extracted_wavelength, extracted_flux, kind='linear', fill_value="extrapolate")
    new_wavelength_grid = np.linspace(center - window_size, center + window_size, num_points)
    resampled_flux = f_interp(new_wavelength_grid)

    return resampled_flux

def create_dataset(wavelength, flux, confirmed_pcygni, num_background=5, window_size=50, num_points=100):
    profiles = []
    labels = []

    # Positive samples (P Cygni profiles)
    for center in confirmed_pcygni:
        profile = extract_window(wavelength, flux, center, window_size, num_points)
        if profile is not None:
            profiles.append(profile)
            labels.append(1)

    # Negative samples (background)
    min_wl, max_wl = np.min(wavelength), np.max(wavelength)
    
    for _ in range(len(confirmed_pcygni) * num_background):  # Get multiple backgrounds per positive
        # Random center that is far enough from confirmed P Cygni features
        while True:
            random_center = random.uniform(min_wl + window_size, max_wl - window_size)
            # Verifying distance
            if np.min(np.abs(confirmed_pcygni - random_center)) > 2 * window_size:
                break
        
        profile = extract_window(wavelength, flux, random_center, window_size, num_points)
        if profile is not None:
            profiles.append(profile)
            labels.append(0)

    # Convert to arrays
    profiles = np.array(profiles)
    labels = np.array(labels)

    return profiles, labels



def main():

    # P-Cygni Template
    template_wave = np.linspace(-50, 50, 500)
    template_flux = (
        1
        + 1 * np.exp(-0.5 * ((template_wave)/7)**2)    # emission
        - 1 * np.exp(-0.5 * ((template_wave + 10)/5)**2)  # absorption
    )

    ## Normalize template
    template_flux = template_flux - np.mean(template_flux)
    template_flux = template_flux / np.std(template_flux)

    # Normalize observed flux
    flux_norm = flux - np.mean(flux)
    flux_norm = flux_norm / np.std(flux)

    # Cross-correlate
    corr = correlate(flux_norm, template_flux, mode='same')

    wl = []
    fl = []
    lb = []
    id = []
    n = 0
    skip = True
    for i in range(len(flux)):
        if corr[i] >= 30.0:
            if skip == False:
                n += 1
                skip = True
            wl.append(wavelength[i])
            fl.append(flux[i])
            lb.append(1)
            id.append(n) 
        else:
            if skip == True:
                n += 1
                skip = False
            wl.append(wavelength[i])
            fl.append(flux[i])
            lb.append(0)
            id.append(n) 

    points = np.column_stack((wl,fl,lb,id))

    # --- Identify strong matches ---
    threshold = 30
    matches = wavelength[corr > threshold]

    peaks, _ = find_peaks(flux_norm, prominence=0.01)
    dips, _ = find_peaks(-flux_norm, prominence=1)

    confirmed_pcygni = []

    for m in matches:
        idx = np.argmin(np.abs(wavelength - m))  # nearest index to match

        # Defining a window around the identified profile
        window = (wavelength > wavelength[idx] - 50) & (wavelength < wavelength[idx] + 50)

        local_peaks = peaks[(wavelength[peaks] > wavelength[idx] - 10) & (wavelength[peaks] < wavelength[idx] + 30)]
        local_dips = dips[(wavelength[dips] > wavelength[idx] - 30) & (wavelength[dips] < wavelength[idx] - 5)]

        #Verify peaks to confirm P-Cygni
        if len(local_peaks) > 0 and len(local_dips) > 0:
            confirmed_pcygni.append(m)

    confirmed_pcygni = np.sort(np.array(confirmed_pcygni))
    final_centers = []
    min_separation = 30

    last_center = -1e9
    for c in confirmed_pcygni:
        if np.abs(c - last_center) > min_separation:
            final_centers.append(c)
            last_center = c

    final_centers = np.array(final_centers)

    profiles, labels = create_dataset(wavelength, flux, final_centers)
    
    for i in range(len(profiles)):
        if labels[i] == 1:
            prf.append(profiles[i])
            lbl.append(1)
        if labels[i] == 0:
            prf.append(profiles[i])
            lbl.append(0)

folder_path = './Filtered/' #Path to raw data
prf = []
lbl = []


# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.fit') or filename.endswith('.fits'):
        full_path = os.path.join(folder_path, filename)
        
        print(f"Opening {full_path}...")
        data = fits.open(full_path)

        crval = data[0].header['CRVAL1']
        cdelt = data[0].header['CDELT1']
        naxis = data[0].header['NAXIS1']

        wavelength = crval + cdelt * np.arange(naxis)
        flux = data[0].data


        
        file = filename
        main()

        data.close()

for i in prf:
    print(len(i))
np.save('profiles', prf)
np.save('labels', lbl)