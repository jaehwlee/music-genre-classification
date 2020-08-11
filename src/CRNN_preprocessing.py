import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)

# Parameters
SONG_SAMPLES = 660000
GTZAN_DIR = '/Users/jaehwan/Desktop/GTZAN/genres/'
GENRES = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

X_TRAIN_PATH = '/Users/jaehwan/Desktop/1396X_train.npy'
Y_TRAIN_PATH = '/Users/jaehwan/Desktop/1396y_train.npy'
X_VALID_PATH = '/Users/jaehwan/Desktop/1396X_valid.npy'
Y_VALID_PATH = '/Users/jaehwan/Desktop/1396y_valid.npy'
X_TEST_PATH = '/Users/jaehwan/Desktop/1396X_test.npy'
Y_TEST_PATH = '/Users/jaehwan/Desktop/1396y_test.npy'

def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]

    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS) ** 2)
    ret = ret[:, :, np.newaxis]
    return ret


def read_data(src_dir, genres):
    # Empty array of dicts with the processed features from all files
    arr_fn = []
    arr_genres = []
    melgrams = np.zeros((0, 96, 1366, 1))
    # Get file list from the folders
    for x, _ in genres.items():
        folder = src_dir + x
        for root, subdirs, files in os.walk(folder):
            for file in files:
                file_name = folder + "/" + file
                S = compute_melgram(file_name)
                S = np.array(list(compute_melgram(file_name)))
                arr_fn.append(S)
                arr_genres.append(genres[x])


    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        arr_fn, arr_genres, test_size=0.2, random_state=42, stratify=arr_genres
    )

    # # Split into train and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    #return X_train, X_test, y_train, y_test
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# Read the data
print('start')
X_train, X_valid, X_test, y_train, y_valid, y_test = read_data(GTZAN_DIR, GENRES)
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)
np.save(X_TRAIN_PATH, X_train)
np.save(Y_TRAIN_PATH, y_train)
np.save(X_TEST_PATH, X_test)
np.save(Y_TEST_PATH, y_test)
np.save(X_VALID_PATH, X_valid)
np.save(Y_VALID_PATH, y_valid)

print('save complete!')
