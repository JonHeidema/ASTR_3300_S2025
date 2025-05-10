import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

from PIL import Image
import io

profiles = np.load('./profiles.npy')
labels = np.load('./labels.npy')

bad  = []
good = []

#splitting our profiles by their labels
for i in range(len(labels)):
    if labels[i] == 0:
        bad.append(profiles[i])
    if labels[i] == 1:
        good.append(profiles[i])


#Creating a template P-Cygni
template_wave = np.linspace(-50, 50, 100)
template_flux = (
    1
    + 1 * np.exp(-0.5 * ((template_wave)/7)**2)    # emission
    - 1 * np.exp(-0.5 * ((template_wave + 10)/5)**2)  # absorption
)

#normalize the template P-Cygni
template_flux = template_flux - np.mean(template_flux)
template_flux = template_flux / np.std(template_flux)

corr = []

for i in good:
    corr.append(correlate(i, template_flux, mode = 'same'))

n = 0
pure   = []
purelab= []
better = []
newlab = []

#Checking each 'good' profile's coorelation factor
for i in range(len(good)):
    if corr[i][50] > 30:
        n += 1
        better.append(good[i])
        newlab.append(1)
        pure.append(good[i])
        purelab.append(1)
print(n)

for i in range(len(bad)):
    better.append(bad[i])
    newlab.append(0)

np.save('data', better)
np.save('reducedLabels', newlab)
np.save('puredata', pure)
np.save('purelab', purelab)

print('Data refined.')