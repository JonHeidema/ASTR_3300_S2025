import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load the profiles and labels
X = np.load('data.npy')
y = np.load('reducedLabels.npy')

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

def sliding_window_predict(spectrum_flux, model, window_size=100, step_size=1):
    predictions = []
    positions = []

    for start in range(0, len(spectrum_flux) - window_size + 1, step_size):
        window = spectrum_flux[start:start+window_size]
        window = window.reshape(1, -1)
        pred = model.predict(window)
        predictions.append(pred[0])
        positions.append(start)
    
    return np.array(predictions), np.array(positions)

def group_detections(predictions, positions, window_size, step_size):
    grouped_positions = []

    is_in_positive = False
    for pred, pos in zip(predictions, positions):
        if pred == 1 and not is_in_positive:
            # Start of a new detection
            grouped_positions.append(pos)
            is_in_positive = True
        elif pred == 0:
            # End of current positive cluster
            is_in_positive = False

    return np.array(grouped_positions)

def plot_detections(wavelength, flux, grouped_positions, window_size):
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, flux, label="Spectrum", color='black')

    for pos in grouped_positions:
        center = wavelength[pos + window_size // 2]
        plt.axvline(center, color='red', linestyle='--', alpha=0.7, label = 'P-Cygni Prediction (Å): %s' % int(center))

    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum with Detected Profiles')
    plt.legend()
    plt.grid()
    plt.show()

# Loading Example Data
spectrum_flux = fits.open('asdb_novaper2020_20210320_834.fit')[0]
crval = spectrum_flux.header['CRVAL1']
cdelt = spectrum_flux.header['CDELT1']
naxis = spectrum_flux.header['NAXIS1']
Xgrid = crval + cdelt * np.arange(naxis)
wavelength = np.linspace(Xgrid[0], Xgrid[-1], len(spectrum_flux.data))

# Predict
window_size = 100
step_size = 1
predictions, positions = sliding_window_predict(spectrum_flux.data, clf, window_size=window_size, step_size=step_size)

# Group detections
grouped_positions = group_detections(predictions, positions, window_size, step_size)


plot_detections(wavelength, spectrum_flux.data, grouped_positions, window_size)

y_score = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # random chance line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"Training accuracy: {clf.score(X_train, y_train)}")
print(f"Testing accuracy: {clf.score(X_test, y_test)}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Results
print(f"K Cross-Validation Accuracies: {accuracies}")
print(f"K Average CV Accuracy: {np.mean(accuracies):.4f}")

#47k training values

#k fold 37953 training, 9489 testing