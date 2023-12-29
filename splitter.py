import time
import logging

import librosa
from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine

logging.basicConfig(format="[%(levelname)s] %(asctime)s %(funcName)s: %(message)s")
logger = logging.getLogger("splitter")
logger.setLevel(logging.INFO)

def init(audio_filename):
    logger.info("Loading " + audio_filename + "...")
    load_start = time.time()

    y, sr = librosa.load(audio_filename, sr=22050)

    load_end = time.time()
    logger.info("Loaded " + audio_filename + " in " + str(round(load_end-load_start, 1)) + "s")
    logger.info("Num. Samples: " + str(len(y)))
    logger.info("Sampling Rate: " + str(sr))

    return y, sr

def main():
    y, sr = init("mix.mp3")

    logger.info("Computing RMS...")
    rms = librosa.feature.rms(y=y)
    norm_rms = rms / np.max(rms)
    norm_rms = norm_rms[0]
    rms_frames = range(len(norm_rms))
    rms_times = librosa.frames_to_time(rms_frames, sr=sr)

    plot(rms_times, norm_rms, "Normalised RMS",
         "Time (s)", "Normalised RMS")
    plt.savefig("normalised_rms.png")

    logger.info("Smoothing RMS...")
    window_size = 200
    smoothed_rms = rollingMax(norm_rms, window_size)
    smoothed_rms_times = rms_times[(window_size-1):(window_size+len(smoothed_rms))]

    logger.info("Finding peaks...")
    peaks = find_peaks(smoothed_rms*(-1), distance=2500, prominence=0.1)[0]
    peak_ts = [((point+window_size) * 512 + 2048/2) / sr for point in peaks]

    plot(smoothed_rms_times, smoothed_rms, "Smoothed Normalised RMS",
         "Time (s)", "Smoothed Normalised RMS")
    plt.xticks(np.arange(0, 2500, 300)) # TODO: Remove when not using test mix.mp3 data
    plt.scatter(peak_ts, smoothed_rms[peaks], color="r")
    plt.savefig("smooth_rms.png")

    slices = getSlices(y, peaks, window_size)
    slices = mergeSlices(slices, sr)
    exportSlices(slices, sr)

def exportSlices(slices, sr):
    logger.info("Exporting slices...")

    original_mix = AudioSegment.from_mp3("mix.mp3")
    cumulative_ms = 0
    for i, s in enumerate(slices):
        logger.debug("Exporting " + str(i+1) + "/" + str(len(slices)) + "...")
        filename = "Part-" + str(i+1) + ".mp3"

        slice_start_ms = cumulative_ms
        slice_end_ms = cumulative_ms + librosa.get_duration(y=s, sr=sr)*1000
        original_mix[slice_start_ms:slice_end_ms]\
            .export(filename, format="mp3")

        cumulative_ms = slice_end_ms

def extractFeatures(audio_slice, sr):
    # Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=audio_slice, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Extracting Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_slice, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Extracting RMS energy
    rms = librosa.feature.rms(y=audio_slice)
    rms_mean = np.mean(rms.T, axis=0)

    # Combine the features into a single feature vector
    combined_features = np.hstack((mfccs_mean, chroma_mean, rms_mean))
    return combined_features

def mergeSlices(slices, sr):
    logger.info("Extracting features...")
    features = [extractFeatures(slice, sr) for slice in slices]

    merged_slices = []
    current_slice = slices[0]
    current_feature = features[0]

    min_duration = 4*60 # 4min
    max_duration = 12*60 # 12min
    similarity_threshold = 0.95

    logger.info("Merging slices...")
    for i in range(1, len(slices)):
        logger.debug("Processing slice " + str(i) + "/" + str(len(slices)) + "...")

        similarity = 1 - cosine(current_feature, features[i])
        logger.debug(similarity)
        current_duration = librosa.get_duration(y=current_slice, sr=sr)
        duration_if_merged = current_duration + librosa.get_duration(y=slices[i], sr=sr)
        if current_duration < min_duration:
            logger.debug("Merge, because under min_duration")
            # Merge if under min_duration
            current_slice = np.concatenate((current_slice, slices[i]))
            current_feature = extractFeatures(current_slice, sr) # Recalculate features for merged slice
        elif duration_if_merged <= max_duration and similarity > similarity_threshold:
            logger.debug("Merge, because similar")
            # Merge if similar and within duration limit
            current_slice = np.concatenate((current_slice, slices[i]))
            current_feature = extractFeatures(current_slice,sr) # Recalculate features for the merged slice
        else:
            if similarity < similarity_threshold:
                logger.debug("Move on, because not similar")
            else:
                logger.debug("Move on, because over max_duration")
            # If not similar or duration exceeded, add to merged slices and move to next
            merged_slices.append(current_slice)
            current_slice = slices[i]
            current_feature = features[i]

    merged_slices.append(current_slice)  # Add the last slice
    return merged_slices

def getSlices(y, peaks, window_size):
    logger.info("Generating slices...")
    slices = []

    prev_yidx = None
    for peak in peaks:
        y_idx = round((peak+window_size) * 512 + 2048/2)
        if prev_yidx is None:
            slices.append(y[0:y_idx])
        else:
            slices.append(y[prev_yidx:y_idx])
        prev_yidx = y_idx+1
    slices.append(y[prev_yidx:])

    return slices

def calibratePeaks(peaks, window_size):
    for peak in peaks:
        calibration_radius = 2.5*60*sr # 2m30s
        calibration_center = (peak+window_size) * 512 + 2048/2
        calibration_start = max(0, round(calibration_center - calibration_radius))
        calibration_end = min(len(y), round(calibration_center + calibration_radius))
        calibration_slice = y[calibration_start:calibration_end]
        # TODO...

def logSliceTimes(slices, sr):
    cumulative_time = 0
    for s in slices:
        slice_time = len(s)/sr
        cumulative_time += slice_time
        logger.debug(f"{round(cumulative_time//60)}m{round(cumulative_time%60)}s")

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
logger.info("Finished!")
