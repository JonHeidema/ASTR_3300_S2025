import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import shutil

path = 'Incoming'

n = 0
i = 0

#Iterating over every file in the folder
for file in os.listdir(path):
    tag = fits.open('incoming/%s' % file)
    crval = tag[0].header['CRVAL1']
    cdelt = tag[0].header['CDELT1']
    naxis = tag[0].header['NAXIS1']
    wavelength = crval + cdelt * np.arange(naxis)
    min = np.min(wavelength)
    max = np.max(wavelength)
    #Verifying a full enough spectrum (atleast 3000 Angstrom wide)
    if file.endswith( '.fit' ) and (max - min) > 3000:
        shutil.copy('Incoming/%s' % file, 'Filtered/%s' % file)
        n += 1
    else: i += 1

print("Files that fit: ", n)
print("Files that don\'t fit: ", i)