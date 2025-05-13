import numpy as np
import matplotlib.pyplot as plt

import bilby

from gwpy.timeseries import TimeSeries


def get_H1(time_of_event, post_trigger_duration, duration, psd_duration_multi):

    H1 = bilby.gw.detector.get_empty_interferometer("H1")

    #### Definite times in relation to the trigger time (time_of_event), duration and post_trigger_duration
    analysis_start = time_of_event + post_trigger_duration - duration
    print("Analysis start time:", analysis_start - time_of_event)
    print("Data segment:", analysis_start - time_of_event, analysis_start + duration - time_of_event)
    #### Use gwpy to fetch the open data
    H1_analysis_data = TimeSeries.fetch_open_data(
    "H1", analysis_start, analysis_start + duration, sample_rate=4096, cache=True)


    #### Initializing the interferometer with strain data
    H1.set_strain_data_from_gwpy_timeseries(H1_analysis_data)
    

    #### Downloading the Power Spectral Data
    psd_duration = duration * psd_duration_multi #32
    psd_start_time = analysis_start - psd_duration
    print("PSD start time:", psd_start_time - time_of_event)
    print("PSD segment:", psd_start_time - time_of_event, psd_start_time + psd_duration - time_of_event)
    H1_psd_data = TimeSeries.fetch_open_data(
    "H1", psd_start_time, psd_start_time + psd_duration, sample_rate=4096, cache=True)


    #### Specifying PSD by proper windowing using psd_alpha used in gwpy
    psd_alpha = 2 * H1.strain_data.roll_off / duration
    print("PSD alpha:", psd_alpha)
    H1_psd = H1_psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")


    #### Now Initializing the interferometer with PSD
    H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=H1_psd.frequencies.value, psd_array=H1_psd.value)


    #### Neglcting the high frequency part at it's a downsampling effect as we are using 4096 Hz
    H1.maximum_frequency = 1024
    print("Neglecting the high frequency part at", H1.maximum_frequency, "Hz as it's a downsampling effect as we are using 4096 Hz data.")


    return H1