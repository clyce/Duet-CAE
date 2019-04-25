import scipy.ndimage as ndimage
import numpy as np


class Noisifier:
    def __init__(self):
        pass

    def noisify_(self, data):
        pass

    def noisify(self, amount, data):
        data_expanded = np.repeat(data, amount, axis=0)
        t = np.array([self.noisify_(d) for d in data_expanded])
        np.random.shuffle(t)
        noised = t[:, 0]
        data = t[:, 1]
        print(noised.shape)

        return np.array([noised, data])


class MnistNoisifier(Noisifier):
    def __init__(self,
                 shift_sd=3,
                 rotate_sd=14,
                 blur_sd=0.3,
                 black_erase_sd=2,
                 white_erase_sd=2,
                 channels=1):
        self.shift_sd = shift_sd
        self.rotate_sd = rotate_sd
        self.blur_sd = blur_sd
        self.black_erase_sd = black_erase_sd
        self.white_erase_sd = white_erase_sd
        self.channels = channels

    def noisify_(self, data):
        rotate = np.random.normal(scale=self.rotate_sd)
        shift = np.random.normal(scale=self.shift_sd, size=(2,))
        blur = np.random.normal(scale=self.blur_sd)
        black_erase = int(np.random.normal(scale=self.black_erase_sd))
        white_erase = int(np.random.normal(scale=self.white_erase_sd))
        noised = ndimage.rotate(data, rotate)
        noised = ndimage.shift(noised, shift)
        noised = ndimage.gaussian_filter(noised, blur)
        if (black_erase >= 1):
            if (np.random.rand() > 0.5):
                i = np.random.randint(0, data.shape[0])
                noised[i:i+black_erase] = 0
            else:
                i = np.random.randint(0, data.shape[1])
                noised[:, i:i+black_erase] = 0
        if (white_erase >= 1):
            m = np.max(data)
            if (np.random.rand() > 0.5):
                i = np.random.randint(0, data.shape[0])
                noised[i:i+white_erase] = m
            else:
                i = np.random.randint(0, data.shape[1])
                noised[:, i:i+white_erase] = m

        return np.array([noised[:data.shape[0], :data.shape[1]], data])


class MotionNoisifier(Noisifier):
    def __init__(self,
                 black_erase_width_sd=8,
                 white_erase_width_sd=8,
                 black_erase_count_sd=3,
                 white_erase_count_sd=5,
                 total_strength_shift_sd=0.5,
                 random_strength_shift_sd=0.3,
                 time_shift_sd=10,
                 ):
        self.black_erase_width_sd = black_erase_width_sd
        self.white_erase_width_sd = white_erase_width_sd
        self.black_erase_count_sd = black_erase_count_sd
        self.white_erase_count_sd = white_erase_count_sd
        self.total_strength_shift_sd = total_strength_shift_sd
        self.random_strength_shift_sd = random_strength_shift_sd
        self.time_shift_sd = time_shift_sd

    def noisify_(self, data):
        total_strength_shift = \
            np.random.normal(scale=self.total_strength_shift_sd)
        noised = np.array(data)
        noised += total_strength_shift
        random_strength_shift = \
            np.random.normal(
                scale=self.random_strength_shift_sd, size=data.shape)
        noised += random_strength_shift
        blackcount = int(abs(np.random.normal(
            scale=self.black_erase_count_sd)))
        whitecount = int(abs(np.random.normal(
            scale=self.white_erase_count_sd)))
        datalen = data.shape[0]
        for _ in range(blackcount):
            width = 1 + \
                int(abs(np.random.normal(scale=self.black_erase_width_sd)))
            i = np.random.randint(0, datalen)
            noised[i: i+width] = 0
        for _ in range(whitecount):
            width = 1 + \
                int(abs(np.random.normal(scale=self.white_erase_width_sd)))
            i = np.random.randint(0, datalen)
            noised[i: i+width] = -1. if (np.random.rand() < 0.5) else 1.

        noised = np.repeat(noised, 3)
        shift = int(np.random.normal(scale=self.time_shift_sd))
        return np.array([noised[shift+datalen:shift+2*datalen], data])


class TextNoisifier(Noisifier):
    def __init__(self,
                 permutation_width_sd=5,
                 permutation_count_sd=10):
        self.permutation_width_sd = permutation_width_sd
        self.permutation_count_sd = permutation_count_sd

    def noisify_(self, data):
        pcount = int(abs(np.random.normal(scale=self.permutation_count_sd)))
        noised = np.array(data)
        for _ in range(pcount):
            pwidth = int(abs(np.random.normal(
                scale=self.permutation_count_sd))) + 2
            pindex = np.random.randint(0, data.shape[0]-pwidth)
            noise_ = np.random.permutation(data[pindex:pindex+pwidth])
            noised[pindex:pindex+pwidth] = noise_
        return np.array([noised, data])
