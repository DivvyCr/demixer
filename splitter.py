import time
import logging

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(asctime)s %(funcName)s: %(message)s")

def init(audio_filename):
    logging.info("Loading " + audio_filename + "...")
    load_start = time.time()

    y, sr = librosa.load(audio_filename, sr=22050)

    load_end = time.time()
    logging.info("Loaded " + audio_filename + " in " + str(round(load_end-load_start, 1)) + "s")
    logging.info("Num. Samples: " + str(len(y)))
    logging.info("Sampling Rate: " + str(sr))

    return y, sr

def main():
    y, sr = init("mix.mp3")

    logging.info("Computing RMS...")
    rms = librosa.feature.rms(y=y)
    norm_rms = rms / np.max(rms)
    norm_rms = norm_rms[0]
    rms_frames = range(len(norm_rms))
    rms_times = librosa.frames_to_time(rms_frames, sr=sr)

    plot(rms_times, norm_rms, "Normalised RMS",
         "Time (s)", "Normalised RMS")
    plt.savefig("normalised_rms.png")

    logging.info("Smoothing RMS...")
    window_size = 200
    smoothed_rms = rollingMax(norm_rms, window_size)
    smoothed_rms_times = rms_times[(window_size-1):(window_size+len(smoothed_rms))]

    logging.info("Finding peaks...")
    peaks = find_peaks(smoothed_rms*(-1), distance=2500, prominence=0.1)[0]
    peak_ts = [((point+window_size) * 512 + 2048/2) / sr for point in peaks]

    plot(smoothed_rms_times, smoothed_rms, "Smoothed Normalised RMS",
         "Time (s)", "Smoothed Normalised RMS")
    plt.scatter(peak_ts, smoothed_rms[peaks], color="r")
    plt.savefig("smooth_rms.png")

# See: https://stackoverflow.com/a/43335059
def rollingMax(a, window):
  def eachValue():
    w = a[:window].copy()
    m = w.max()
    yield m
    i = 0
    j = window
    while j < len(a):
      oldValue = w[i]
      newValue = w[i] = a[j]
      if newValue > m:
        m = newValue
      elif oldValue == m:
        m = w.max()
      yield m
      i = (i + 1) % window
      j += 1
  return np.array(list(eachValue()))

def plot(xs, ys, title, xlabel, ylabel):
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, color="b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

main()
logging.info("Finished!")
