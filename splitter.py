import argparse
import logging
import os
import subprocess
import sys

import librosa
from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean

parser = argparse.ArgumentParser(prog="Splitter", description="Split your DJ mixes!")
parser.add_argument("url",
                    help="A YouTube link to the mix. It will be downloaded and split")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Set the logging level to INFO")
parser.add_argument("-vv", "--very-verbose", action="store_true",
                    help="Set the logging level to DEBUG")
parser.add_argument("--make-plots", action="store_true",
                    help="Generate plots (ie. graphs) for debugging")
parser.add_argument("--no-slices", action="store_true",
                    help="Do not generate slices. Only do mix analysis")
parser.add_argument("--no-export", action="store_true",
                    help="Do not export slices. Only do mix analysis and slice generation")
args = parser.parse_args()

logging.basicConfig(format="[%(levelname)s] %(asctime)s %(funcName)s: %(message)s")
logger = logging.getLogger("splitter")
if args.verbose:
    logger.setLevel(logging.INFO)
elif args.very_verbose:
    logger.setLevel(logging.DEBUG)

audio_filename = "mix.mp3"
if os.path.exists(audio_filename):
    print("Deleting mix.mp3...")
    os.remove(audio_filename)

print("Downloading " + args.url + " ...")
ytdlp_args = ["yt-dlp", args.url,
              "--extract-audio", "--audio-format", "mp3",
              "--output", audio_filename,
              ("--quiet" if logger.level == 0 else "--no-quiet")]
subprocess.Popen(ytdlp_args).wait()

print("Loading " + audio_filename + "...")
y, sr = librosa.load(audio_filename, sr=22050)
logger.info("Num. Samples: " + str(len(y)))
logger.info("Sampling Rate: " + str(sr))

def main():
    print("Processing audio...")

    logger.info("Computing RMS...")
    rms = librosa.feature.rms(y=y)
    norm_rms = rms / np.max(rms)
    norm_rms = norm_rms[0]
    rms_frames = range(len(norm_rms))
    rms_times = librosa.frames_to_time(rms_frames, sr=sr)

    if args.make_plots:
        plot(rms_times, norm_rms, "Normalised RMS",
            "Time (s)", "Normalised RMS")
        plt.xticks(np.arange(0, np.max(rms_times), 300))
        plt.savefig("normalised_rms.png")

    logger.info("Smoothing RMS...")
    window_size = 200
    smoothed_rms = rollingMax(norm_rms, window_size)
    smoothed_rms_times = rms_times[(window_size-1):(window_size+len(smoothed_rms))]

    if args.no_slices: quit()

    logger.info("Finding peaks...")
    peaks = approxDerivative(smoothed_rms)
    peak_ts = [((point+window_size) * 512 + 2048/2) / sr for point in peaks]

    if args.make_plots:
        plot(smoothed_rms_times, smoothed_rms, "Smoothed Normalised RMS",
             "Time (s)", "Smoothed Normalised RMS")
        for pt in peak_ts:
            plt.axvline(x=pt, color='r', linestyle='--', zorder=10)
        plt.xticks(np.arange(0, np.max(smoothed_rms_times), 300))
        plt.savefig("smooth_rms.png")


    slices = getSlices(peaks, window_size)
    slices = mergeSlices(slices)
    if args.make_plots:
        plotSliceTimes(slices, smoothed_rms_times, smoothed_rms)

    if args.no_export: quit()
    exportSlices(slices)

def exportSlices(slices):
    print("Input metadata for export:")
    mix_title =  input(" Mix Title  > ")
    mix_artist = input(" Mix Artist > ")
    mix_year =   input(" Mix Year   > ")

    print("Loading " + audio_filename + " for export...")
    original_mix = AudioSegment.from_mp3("mix.mp3")

    print("Exporting slices...")
    cumulative_ms = 0
    for i, s in enumerate(slices):
        if logger.level > logging.DEBUG or logger.level == 0:
            updateProgressBar(i, len(slices))

        filename = "Part-" + str(i+1) + ".mp3"

        slice_start_ms = cumulative_ms
        slice_end_ms = cumulative_ms + librosa.get_duration(y=s, sr=sr)*1000
        original_mix[slice_start_ms:slice_end_ms]\
            .export(filename, format="mp3",
                    tags={"TITLE": ("Part " + str(i+1)),
                          "ALBUM": mix_title,
                          "ALBUMARTIST": mix_artist,
                          "YEAR": mix_year,
                          "GENRE": "DJ Mix",
                          "TRACK": str(i+1),
                          "COMMENT": "NOTE: This is YouTube audio and part of a larger mix."})
        cumulative_ms = slice_end_ms

    if logger.level > logging.DEBUG or logger.level == 0:
        updateProgressBar(len(slices), len(slices))
        sys.stdout.write('\n') # Newline after progress bar

def extractFeatures(audio_slice):
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
    combined_features = np.hstack((mfccs_mean, chroma_mean))
    return combined_features

def mergeSlices(slices):
    logger.info("Extracting features...")
    features = [extractFeatures(slice) for slice in slices]

    sims = []
    for idx in range(len(features)-1):
        sims.append(euclidean(features[idx], features[idx+1]))
    similarity_threshold = np.mean(sims)-0.5*np.std(sims)
    logger.debug("Threshold for similarity: " + str(round(similarity_threshold, 2)))

    min_duration = 4.5*60 # 4m30s
    max_duration = 10*60 # 10m

    merged_slices = []
    current_slice = slices[0]
    current_feature = features[0]

    print("Merging slices...")
    for i in range(1, len(slices)):
        if logger.level > logging.DEBUG or logger.level == 0:
            updateProgressBar(i, len(slices))

        # TODO: Only compare to last ~1-2min instead of whole slice?

        similarity = euclidean(current_feature, features[i])
        logger.debug("Similarity: " + str(round(similarity, 3)))
        current_duration = librosa.get_duration(y=current_slice, sr=sr)
        duration_if_merged = current_duration + librosa.get_duration(y=slices[i], sr=sr)
        if current_duration < min_duration:
            logger.debug("Merge, because under min_duration")
            # Merge if under min_duration
            current_slice = np.concatenate((current_slice, slices[i]))
            current_feature = extractFeatures(current_slice) # Recalculate features for merged slice
        elif duration_if_merged <= max_duration and similarity < similarity_threshold:
            logger.debug("Merge, because similar")
            # Merge if similar and within duration limit
            current_slice = np.concatenate((current_slice, slices[i]))
            current_feature = extractFeatures(current_slice) # Recalculate features for the merged slice
        else:
            if i == (len(slices)-1):
                # The next slice is the last one,
                # so merge it to avoid having a very short slice at the end
                current_slice = np.concatenate((current_slice, slices[i]))
            else:
                if similarity > similarity_threshold:
                    logger.debug("Move on, because not similar")
                else:
                    logger.debug("Move on, because over max_duration")
                merged_slices.append(current_slice)
                current_slice = slices[i]
                current_feature = features[i]


    if logger.level > logging.DEBUG or logger.level == 0:
        updateProgressBar(len(slices), len(slices))
        sys.stdout.write('\n') # Newline after progress bar

    merged_slices.append(current_slice)  # Add the last slice
    return merged_slices

def getSlices(peaks, window_size):
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

def plotSliceTimes(slices, rms_times, rms_values):
    plot(rms_times, rms_values, "Smoothed Normalised RMS",
            "Time (s)", "Smoothed Normalised RMS")
    plt.xticks(np.arange(0, np.max(rms_times), 300))

    cumulative_time = 0
    for s in slices:
        slice_time = len(s)/sr
        cumulative_time += slice_time
        plt.axvline(x=cumulative_time, color='r', linestyle='--', zorder=10)
        # logger.debug(f"{round(cumulative_time//60)}m{round(cumulative_time%60)}s")

    plt.savefig("exported.png")

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

def approxDerivative(a):
    window_size = 300 # Should probably be related to the smoothing window?
    derivative = np.array([a[i + window_size] - a[i] for i in range(len(a) - window_size)])

    mix_duration = librosa.get_duration(y=y, sr=sr) # Seconds
    num_peaks = int(mix_duration / (2*60)) # One peak per 2min
    derivative_second = len(derivative) // mix_duration

    peak_idxs, _ = find_peaks(derivative*(-1), distance=60*derivative_second, prominence=np.std(a))
    logger.debug("Filtering " + str(len(peak_idxs)) + " peaks to " + str(num_peaks) + "...")
    peak_idxs_sorted_by_derivative = peak_idxs[np.argsort(derivative[peak_idxs])]
    peak_idxs_filtered = peak_idxs_sorted_by_derivative[:num_peaks]

    return np.sort(peak_idxs_filtered)[1:] # First index is always near-zero

def updateProgressBar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '#' * int(percent / 5) + ' ' * (20 - int(percent / 5))
    sys.stdout.write("\r[{}] {:.0f}%".format(bar, percent))
    sys.stdout.flush()

def plot(xs, ys, title, xlabel, ylabel):
    plt.figure(figsize=(24, 8))
    plt.plot(xs, ys, color="b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

main()
