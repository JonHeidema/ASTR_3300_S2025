import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from astropy.io import fits
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
import random
import os

#Images that worked:
#./novadel2013mod/asdb_novadel2013_20130814_844.fit
#./novadel2013mod/asdb_novadel2013_20131114_844.fit










def extract_window(wavelength, flux, center, window_size=50, num_points=100):
    """Extract and resample a window centered at a given wavelength."""
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
            labels.append(1)  # Label: P Cygni = 1

    # Negative samples (background)
    min_wl, max_wl = np.min(wavelength), np.max(wavelength)
    
    for _ in range(len(confirmed_pcygni) * num_background):  # Get multiple backgrounds per positive
        # Random center that is far enough from confirmed P Cygni features
        while True:
            random_center = random.uniform(min_wl + window_size, max_wl - window_size)
            # Make sure it's at least 2×window_size away from any P Cygni feature
            if np.min(np.abs(confirmed_pcygni - random_center)) > 2 * window_size:
                break
        
        profile = extract_window(wavelength, flux, random_center, window_size, num_points)
        if profile is not None:
            profiles.append(profile)
            labels.append(0)  # Label: Background = 0

    # Convert to arrays
    profiles = np.array(profiles)
    labels = np.array(labels)

    return profiles, labels



def main():

    # ---- Now create a synthetic P Cygni template ----
    template_wave = np.linspace(-50, 50, 500)  # template window in Angstroms
    template_flux = (
        1
        + 1 * np.exp(-0.5 * ((template_wave)/7)**2)    # emission
        - 1 * np.exp(-0.5 * ((template_wave + 10)/5)**2)  # absorption
    )

    ## Normalize template (important for cross-correlation)
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
    threshold = 30  # You might tune this
    matches = wavelength[corr > threshold]
    #print("Potential P Cygni feature detected at wavelengths:", matches)

    # Find peaks (emission) and dips (absorption) in the flux

    peaks, _ = find_peaks(flux_norm, prominence=0.01)    # tweak prominence depending on noise
    dips, _ = find_peaks(-flux_norm, prominence=1)    # same for dips (negative flux)

    confirmed_pcygni = []

    for m in matches:
        idx = np.argmin(np.abs(wavelength - m))  # nearest index to match

        # Define a local window around the match, say +/- 50 Å
        window = (wavelength > wavelength[idx] - 50) & (wavelength < wavelength[idx] + 50)

        local_peaks = peaks[(wavelength[peaks] > wavelength[idx] - 10) & (wavelength[peaks] < wavelength[idx] + 30)]
        local_dips = dips[(wavelength[dips] > wavelength[idx] - 30) & (wavelength[dips] < wavelength[idx] - 5)]

        if len(local_peaks) > 0 and len(local_dips) > 0:
            # Dip (absorption) is to the blue of peak (emission)
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

    map = np.array(['black', 'red'])

    color = map[lb]

    plt.scatter(wavelength, flux, c = color, s = 2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('P-Cygni Indicated Spectrum')

        # Plot everything
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    plt.axhline(30, c = 'black', linestyle = '--')

    for i in final_centers:
        plt.axvline(i, c = 'grey', linestyle = '--', alpha = 0.5)


    unique_n = np.unique(id)
    for n in unique_n:
        mask = (points[:, 3] == n)
        ax[0].plot(points[:, 0][mask], flux_norm[mask], label=f'Segment {int(n)}')

    #ax[0].plot(wavelength, flux)
    ax[0].set_ylabel("Flux")
    ax[0].set_title("Observed Spectrum")

    ax[1].plot(template_wave + wavelength[len(wavelength)//2], template_flux)
    ax[1].set_ylabel("Template")
    ax[1].set_title("P Cygni Template (shifted)")

    ax[2].plot(wavelength, corr)
    ax[2].set_ylabel("Cross-correlation")
    ax[2].set_xlabel("Wavelength (Å)")
    ax[2].set_title("Cross-correlation output")

    plt.tight_layout()
    plt.show()


    plt.hist(corr[corr>0], bins=1000)
    plt.show()
    #Loading in Spectral Data
    
data = fits.open('./Filtered/asdb_novadel2013_20130814_844.fit')
crval = data[0].header['CRVAL1']
cdelt = data[0].header['CDELT1']
naxis = data[0].header['NAXIS1']

wavelength = crval + cdelt * np.arange(naxis)
flux = data[0].data

main()