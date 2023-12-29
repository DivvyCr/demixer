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
    plt.xticks(np.arange(0, 2500, 300)) # TODO: Remove when not using test mix.mp3 data
    plt.scatter(peak_ts, smoothed_rms[peaks], color="r")
    plt.savefig("smooth_rms.png")

    # for peak in peaks:
    #     calibration_radius = 2.5*60*sr # 2m30s
    #     calibration_center = (peak+window_size) * 512 + 2048/2
    #     calibration_start = max(0, round(calibration_center - calibration_radius))
    #     calibration_end = min(len(y), round(calibration_center + calibration_radius))
    #     calibration_slice = y[calibration_start:calibration_end]
    #     slices.append(calibration_slice)

    slices = []
    prev_yidx = None
    for peak in peaks:
        y_idx = round((peak+window_size) * 512 + 2048/2)
        if prev_yidx is None:
            slices.append(y[0:y_idx])
        else:
            slices.append(y[prev_yidx:y_idx])
        prev_yidx = y_idx+1

    for i in range(3):
        print(i)
        new_slices = mergeMostSimilarSlices(slices, sr)
        if len(new_slices) == len(slices):
            break # No slices merged
        slices = new_slices

    print(len(slices))
    temp = 0
    for s in slices:
        x = len(s)/sr
        temp += x
        print(f"{round(temp//60)}m{round(temp%60)}s")


def extractFeatures(audio_slice, sr):
    # Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=audio_slice, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Extracting Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_slice, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # # Extracting Spectral Contrast features
    # spectral_contrast = librosa.feature.spectral_contrast(y=audio_slice, sr=sr)
    # spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    # Combine the features into a single feature vector
    combined_features = np.hstack((mfccs_mean, chroma_mean))
    return combined_features

def mergeMostSimilarSlices(slices, sr):
    # TODO: Cache feature data, and only update for merged slices!
    features = [extractFeatures(slice, sr) for slice in slices]

    # Find:
    min_distance = float('inf')
    min_index = None
    for i in range(len(features) - 1):
        distance = np.linalg.norm(features[i] - features[i + 1])
        if distance < min_distance:
            min_distance = distance
            min_index = i

    # Merge:
    if min_index is not None:
        merged_slice = np.concatenate((slices[min_index], slices[min_index + 1]))
        new_slices = slices[:min_index] + [merged_slice] + slices[min_index + 2:]
        return new_slices
    else:
        logging.info("No similar slices?")
        return slices

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
