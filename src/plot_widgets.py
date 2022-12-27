import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Figure
import numpy as np
import librosa.display
from matplotlib.patches import ConnectionPatch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import asyncio

class GenericMatPlot(Figure):

    def __init__(self, fig, axis, parent=None):
        self.__fig = fig
        self.__axis = axis
        super(GenericMatPlot, self).__init__(self.__fig)
        self.setParent(parent)

    def __clearAxis(self):
        if type(self.__axis) is np.ndarray:
            for ax in self.__axis:
                ax.cla()
        else:
            self.__axis.cla()
        self.draw()

    def updatePlot(self, x, y, index=-1):
        if index == -1 or (not (type(self.__axis) is np.ndarray)):
            self.__axis.plot(x, y)
            return
        self.__axis[index].cla()
        self.__axis[index].plot(x, y)
        self.draw()


class Spectrogram(GenericMatPlot):

    def __init__(self, parent=None):
        self.__fig, self.__ax = plt.subplots(nrows=2, ncols=1, sharex=False)
        super(Spectrogram, self).__init__(self.__fig, self.__ax, parent)
        self.__f = True

    def makePlot(self, y1, y2, sr, hop_length):
        self.__ax[0].cla()
        self.__ax[1].cla()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time',
                                       sr=sr, ax=self.__ax[0])
        self.__ax[0].set(title='Log-frequency power spectrogram')
        self.__ax[0].label_outer()

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y2, hop_length=hop_length)),
                                    ref=np.max)
        librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                                 x_axis='time', ax=self.__ax[1])
        self.__ax[1].set(title='Log-frequency power spectrogram')
        self.__ax[1].label_outer()
        if self.__f:
            self.__fig.colorbar(img, ax=self.__ax, format="%+2.f dB")
            self.__f = False

        self.draw()


class DTWGraph(GenericMatPlot):

    def __init__(self, parent=None):
        self.__fig, self.__ax = plt.subplots(nrows=2)
        super(DTWGraph, self).__init__(self.__fig, self.__ax, parent)
        self.__f = True

    def makePlot(self, x, y, sr, hop_length):
        self.__ax[0].cla()
        self.__ax[1].cla()
        ref_mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length)
        test_mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        D, wp = librosa.sequence.dtw(X=ref_mfcc, Y=test_mfcc, metric='euclidean')

        wps = librosa.frames_to_time(wp, sr=sr, hop_length=hop_length)
        img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=sr, hop_length=hop_length, ax=self.__ax[0])
        self.__ax[0].plot(wps[:, 1], wps[:, 0], marker='.', color='r')
        if self.__f:
            self.__fig.colorbar(img, ax=self.__ax)
            self.__f = False
        self.__ax[1].plot(D[-1, :] / wp.shape[0])
        self.__ax[0].set_title("Warping Paths")
        self.__ax[1].set_title("Cost")
        self.__ax[0].label_outer()
        self.__ax[1].label_outer()
        self.draw()


class RelationGraph(GenericMatPlot):

    def __init__(self, parent=None):
        self.__fig, self.__ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 4))
        super(RelationGraph, self).__init__(self.__fig, self.__ax, parent)

    def makePlot(self, x, y, sr, hop_length):
        self.__ax[0].cla()
        self.__ax[1].cla()

        ref_mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length)
        test_mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        D, wp = librosa.sequence.dtw(X=ref_mfcc, Y=test_mfcc, metric='euclidean')
        wps = librosa.frames_to_time(wp, sr=sr, hop_length=hop_length)
        # Plot x_2
        librosa.display.waveshow(y, sr=sr, ax=self.__ax[1])
        self.__ax[1].set(title='Reference')

        # Plot x_1
        librosa.display.waveshow(x, sr=sr, ax=self.__ax[0])
        self.__ax[0].set(title='Sample')
        self.__ax[0].label_outer()

        n_arrows = 20
        for tp1, tp2 in wps[::len(wps) // n_arrows]:
            # Create a connection patch between the aligned time points
            # in each subplot
            con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0),
                                  axesA=self.__ax[0], axesB=self.__ax[1],
                                  coordsA='data', coordsB='data',
                                  color='r', linestyle='--',
                                  alpha=0.5)
            self.__ax[1].add_artist(con)

        self.draw()


class FeatureGraph(GenericMatPlot):

    def __init__(self, parent=None):
        self.__fig, self.__ax = plt.subplots(nrows=2, sharey=True)
        super(FeatureGraph, self).__init__(self.__fig, self.__ax, parent)
        self.__f = True

    def makePlot(self, x, y, sr, hop_length):
        self.__ax[0].cla()
        self.__ax[1].cla()
        ref_mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length)
        test_mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)

        img = librosa.display.specshow(ref_mfcc, x_axis="time", y_axis="frames", hop_length=hop_length, ax=self.__ax[0])
        librosa.display.specshow(test_mfcc, x_axis="time", y_axis="frames", hop_length=hop_length, ax=self.__ax[1])
        self.__ax[0].set_title("Reference")
        self.__ax[1].set_title("Testcase")
        self.__ax[0].label_outer()
        self.__ax[1].label_outer()
        if self.__f:
            self.__fig.colorbar(img, ax=self.__ax)
            self.__f = False


class WaveFormFigure(GenericMatPlot):

    def __init__(self, parent=None):
        self.__fig, self.__ax = plt.subplots(nrows=2, sharex=True, sharey=False)
        super(WaveFormFigure, self).__init__(self.__fig, parent)

    def makePlot(self, ref, test, fs):
        self.__ax[0].cla()
        self.__ax[1].cla()

        librosa.display.waveshow(y=ref, sr=fs, ax=self.__ax[0])
        self.__ax[0].set_title("Reference")
        self.__ax[0].label_outer()
        librosa.display.waveshow(y=test, sr=fs, ax=self.__ax[1])
        self.__ax[1].set_title("Test")
        self.__ax[1].label_outer()

        self.draw()
