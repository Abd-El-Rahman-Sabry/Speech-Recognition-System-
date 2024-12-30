from PyQt5.QtCore import QDir, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from playsound import playsound
from pydub.playback import play

from plot_widgets import *
from recognition import WordRecognition
from testcase import *
import os.path
from mic import LiveFFTWidget
from PyQt5.QtCore import Qt
from scipy.io.wavfile import write


class DirSection(QWidget):
    """
    A QWidget section that displays a file tree view of directories.
    This allows users to browse and select directories for processing.
    """
    def __init__(self, parent=None):
        super(DirSection, self).__init__(parent=parent)
        self.setLayout(QHBoxLayout())
        self.__treeView = QTreeView()

        path = os.getcwd()
        print(path)
        self.__dirModel = QFileSystemModel()
        self.__dirModel.setRootPath(path)
        self.__dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)
        self.__treeView.setModel(self.__dirModel)
        self.__treeView.setRootIndex(self.__dirModel.index(path))

        self.layout().addWidget(self.__treeView)
        self.__treeView.clicked.connect(self.connect)
        self.__treeView.installEventFilter(self)

    def connect(self, index):
        """
        Handles the directory selection and updates the parent widget with the selected path.
        """
        file = self.__dirModel.fileInfo(index).filePath()
        self.parent().parent().updatePath(file)

    def eventFilter(self, source, event):
        """
        Handles context menu events for directory tree view.
        """
        if event.type() == QEvent.ContextMenu and source is self.__treeView:

            menu = QMenu()
            menu.addAction('Run')
            menu.addAction('Make Reference')

            action = menu.exec_(event.globalPos())
            if action:
                item = self.__treeView.selectedIndexes()
                file = self.__dirModel.fileInfo(item[0]).filePath()
                if action.text() == 'Run':
                    self.parent().parent().executeTestCase(file)
                elif action.text() == 'Make Reference':
                    ex = re.compile(r'G(\d+)S(\d)(\S)(\d+)[WM]P(\d+)W?(\d)?')
                    a = re.findall(ex, file)
                    if len(a):
                        a = a[0]
                        if os.path.exists(file + "\\Wav"):
                            test_case = Testcase(file)
                            self.parent().parent().wr.setReference(test_case, a['speaker type'])

                print(file)
            return True
        return super().eventFilter(source, event)


class FileSection(QWidget):
    """
    A QWidget section that displays a list of files in the current directory.
    Users can select and interact with files for processing.
    """
    def __init__(self, parent=None):
        super(FileSection, self).__init__(parent=parent)
        self.setLayout(QHBoxLayout())
        self.__listView = QListView()

        path = os.getcwd()
        print(path)
        self.__dirModel = QFileSystemModel()
        self.__dirModel.setRootPath(path)
        self.__dirModel.setFilter(QDir.NoDotAndDotDot | QDir.Files)
        self.__listView.setModel(self.__dirModel)
        # self.__listView.setRootIndex(self.__dirModel.index(path))

        self.layout().addWidget(self.__listView)
        self.__listView.clicked.connect(self.connect)
        self.__listView.installEventFilter(self)

    def setPath(self, path):
        """
        Sets the path for the file section and updates the view.
        """
        self.__dirModel.setRootPath(path)
        self.__listView.setRootIndex(self.__dirModel.index(path))

    def connect(self, index):
        """
        Connects the selected file and performs the relevant operation.
        """
        file = self.__dirModel.fileInfo(index).filePath()

    def eventFilter(self, source, event):
        """
        Handles context menu events for the file list view.
        """
        if event.type() == QEvent.ContextMenu and source is self.__listView:
            menu = QMenu()
            menu.addAction('Run')
            menu.addAction('Play')

            action = menu.exec_(event.globalPos())
            if action:
                item = self.__listView.selectedIndexes()
                file = self.__dirModel.fileInfo(item[0]).filePath()
                if action.text() == 'Run':
                    ex = re.compile(r'G(\d+)S(\d)(\S)(\d+)[WM]P(\d+)W?(\d)?')
                    a = re.findall(ex, file)
                    if len(a):
                        a = a[0]
                        sample = {}
                        sample['student_number'] = int(a[1])
                        sample['speaker type'] = a[2]
                        sample['speaker age'] = int(a[3])
                        sample['word pair'] = int(a[4])
                        sample['word'] = int(a[5]) if a[5] != '' else 1
                        t, f = librosa.load(file, sr=None)
                        sample['wav'] = t
                        self.parent().parent().exeFile(sample)
                elif action.text() == 'Play':
                    self.parent().parent().playSound(file)

            return True
        return super().eventFilter(source, event)


class MainApp(QMainWindow):
    """
    Main application class for the SSA Sound Recognition System.
    This class manages the user interface, audio processing, and word recognition.
    """
    WordsList = [
        'ذا', "ثا", "ثا", "سا", "نذير", "نظير", "تين", "طين", "دل", "ضل", "عد", "عض", 
        "صامد", "صامت", "ضن", "ظن", "لمس", "لمز", "مسح", "مسخ", "مصير", "مسير", "ذل", 
        "زل", "زفر", "ظفر", "خير", "غير", "قلب", "كلب", "خائن", "كائن", "مسك", "مسخ", 
        "علم", "ألم", "حمزة", "همزة", "حالة", "هالة", "حرم", "هرم", "حامل", "خامل", 
        "هزم", "حزم", "الحمدلله"
    ]

    def __init__(self):
        super(MainApp, self).__init__(None)
        l = len(MainApp.WordsList)

        for i in range(0, l - 1, 2):
            print(MainApp.WordsList[i], MainApp.WordsList[i + 1])
        self.__currentTimeWaveFrom = 0

        self.__currentRef = 0
        self.__makeWordRecognition()
        self.__dirSide = DirSection(self)
        self.__fileSide = FileSection(self)
        self.__tabView = QTabWidget(self)
        self.__makeDockWidget("Dires", self.__dirSide, Qt.LeftDockWidgetArea)
        self.__makeDockWidget("Files", self.__fileSide, Qt.LeftDockWidgetArea)
        self.__log = QTextEdit(self)
        self.__log.setReadOnly(True)
        self.__log.append("\n Welcome to SSA Sound Recognition System\n")
        self.__log.append(" all rights reserved © 2022 . Sohila . Sabry . Awwad\n")
        self.__makeDockWidget("Logging", self.__log, Qt.BottomDockWidgetArea)
        self.setCentralWidget(self.__tabView)
        self.__makeFigures()
        self.__makeRecorder()
        self.__addTab(self.__mainRecorder, "Recorder")

        # General Variables
        self.__currentGender = 'M'
        self.__currentListIndex = 0

    def __makeWordRecognition(self):
        """
        Initializes the word recognition system.
        """
        self.wr = WordRecognition()
        self.__hop_length = 512

    def updatePath(self, path):
        """
        Updates the file section with the new directory path.
        """
        self.__fileSide.setPath(path)

    def executeTestCase(self, path):
        """
        Executes a test case for a given directory path and updates the logs and plots.
        """
        if os.path.exists(path + "\\Wav"):
            test_case = Testcase(path)
            g, t = self.wr.decide_gender(test_case)
            types = {"C": "Child", "F": "Female", "M": "Male"}
            self.__log.append(
                f'\n The selected path has gender {types[g]} and the right gender is {types[test_case.get_main_sample()["speaker type"]]}')
            print(f"This shit is here {test_case.get_main_sample()}")
            self.__currentTimeWaveFrom = test_case.get_main_sample()['wav']
            self.__currentRef = t
            try:
                self.__updatePlots()
            except Exception as e:
                self.__log.append(f'\n {e}')

    def exeFile(self, sample):
        """
        Executes a file sample for recognition and updates the plots.
        """
        t = sample['wav']
        index = (sample['word pair'] - 1) * 2 + sample['word'] - 1
        g = sample['speaker type']
        ref = self.wr.getReference(g , index)
        self.__currentRef = ref['wav']
        self.__currentTimeWaveFrom = t
        self.__updatePlots()

    def __updatePlots(self):
        """
        Updates the plots for the recognition system.
        """
        self.__spectorgram.makePlot(self.__currentTimeWaveFrom, self.__currentRef, WordRecognition.SampleRate, self.__hop_length)
        self.__dtwGraph.makePlot(self.__currentTimeWaveFrom, self.__currentRef, WordRecognition.SampleRate, self.__hop_length)
        self.__relationGraph.makePlot(self.__currentTimeWaveFrom, self.__currentRef, WordRecognition.SampleRate, self.__hop_length)
        self.__featureGraph.makePlot(self.__currentTimeWaveFrom, self.__currentRef, WordRecognition.SampleRate, self.__hop_length)
        self.__waveGraph.makePlot(self.__currentTimeWaveFrom, self.__currentRef, WordRecognition.SampleRate)

    def __makeRecorder(self):
        """
        Creates the audio recorder UI and associated controls.
        """
        self.__mainRecorder = QWidget(self)
        self.__mainRecorder.setLayout(QVBoxLayout())
        self.__controlButtons = QWidget(self)
        self.__mainRecorder.layout().addWidget(self.__controlButtons)
        self.__controlButtons.setLayout(QHBoxLayout())

        # Buttons
        self.__play = QPushButton("Record", clicked=self.__startRecord)
        self.__play.setIcon(QIcon('icons/mic.png'))
        self.__pause = QPushButton("Pause", clicked=self.__pauseRecord)
        self.__pause.setIcon(QIcon('icons/pause.png'))
        self.__stop = QPushButton("Stop", clicked=self.__finishRecord)
        self.__stop.setIcon(QIcon('icons/stop.png'))
        self.__run = QPushButton("Play", clicked=self.__playRecord)
        self.__run.setIcon(QIcon('icons/play.png'))
        self.__check = QPushButton("Check G", clicked=self.__check)
        self.__check.setIcon(QIcon('icons/check.png'))
        self.__checkP = QPushButton("Check P", clicked=self.__checkP)
        self.__checkP.setIcon(QIcon('icons/check.png'))
        self.__checkList = QComboBox(self)
        self.__checkList.addItems(MainApp.WordsList)
        self.__checkList.currentIndexChanged.connect(self.__indexChanged)

        # Layout for control buttons
        self.__controlButtons.layout().addWidget(self.__play)
        self.__controlButtons.layout().addWidget(self.__pause)
        self.__controlButtons.layout().addWidget(self.__stop)
        self.__controlButtons.layout().addWidget(self.__run)
        self.__controlButtons.layout().addWidget(self.__check)
        self.__controlButtons.layout().addWidget(self.__checkP)
        self.__controlButtons.layout().addWidget(self.__checkList)

        # Recorder Graphs
        self.__recoder = LiveFFTWidget()
        self.__mainRecorder.layout().addWidget(self.__recoder)

    def __indexChanged(self, ind):
        """
        Updates the current word pair index when the selection changes.
        """
        print(ind)
        self.__currentListIndex = ind

    def __checkP(self):
        """
        Checks the recorded sound and updates the logs with the recognition result.
        """
        s, t = librosa.load('temp.wav', sr=None)
        r = self.wr.decide_speech_pair(s, self.__currentGender, self.__currentListIndex)
        self.__currentTimeWaveFrom = s
        self.__currentRef = self.wr.getReference(self.__currentGender, self.__currentListIndex)['wav']
        self.__updatePlots()
        if r == 1:
            self.__log.append("\n Right")
        elif r == 0:
            self.__log.append("\n Wrong")
        else:
            self.__log.append("\n Others")

    def __makeDockWidget(self, name, widget, side):
        """
        Creates a dock widget with a specified name, widget, and position.
        """
        doc = QDockWidget(name, self)
        doc.setWidget(widget)
        self.addDockWidget(side, doc)

    def __makeFigures(self):
        """
        Creates and adds various plots for visualizing the recognition process.
        """
        self.__spectorgram = Spectrogram(self)
        self.__addTab(self.__spectorgram, "Spectrogram")
        self.__dtwGraph = DTWGraph(self)
        self.__addTab(self.__dtwGraph, "DTW")
        self.__relationGraph = RelationGraph(self)
        self.__addTab(self.__relationGraph, "Connections")
        self.__featureGraph = FeatureGraph(self)
        self.__addTab(self.__featureGraph, "MFCC Features")
        self.__waveGraph = WaveFormFigure(self)
        self.__addTab(self.__waveGraph, "WaveForms")

    def __addTab(self, widget, title):
        """
        Adds a tab to the main tab view.
        """
        self.__tabView.addTab(widget, title)

    def __startRecord(self):
        """
        Starts the audio recording process.
        """
        self.__run.setDisabled(True)
        self.__recoder.mic.start_recording()
        self.__log.append('\n Recording Now')

    def __pauseRecord(self):
        """
        Pauses the audio recording process.
        """
        self.__run.setDisabled(False)
        self.__recoder.mic.pause_recording()
        self.__log.append('\n Recording is paused')

    def __finishRecord(self):
        """
        Finishes the audio recording process.
        """
        self.__run.setDisabled(False)
        self.__recoder.mic.end_recording()
        self.__log.append('\n Done !! Record saved')

    def __playRecord(self):
        """
        Plays back the recorded audio.
        """
        if os.path.exists('temp.wav'):
            self.playSound('temp.wav')
        else:
            self.__log.append("Can't find the recorded Sound")

    def __check(self):
        """
        Checks the recorded sound for gender recognition and updates the logs.
        """
        if os.path.exists('temp.wav'):
            t, f = librosa.load('temp.wav')
            g, ref = self.wr.decide_gender(t)
            types = {"C": "Child", "F": "Female", "M": "Male"}
            self.__log.append(f'\n Your Gender is {types[g]}')
            self.__currentTimeWaveFrom = t
            self.__currentRef = ref
            self.__updatePlots()
        else:
            self.__log.append("Can't find the recorded Sound")

    def playSound(self, path: str):
        """
        Plays the sound file at the given path.
        """
        if path.find('.wav') != -1:
            audio = AudioSegment.from_wav(path)
        else:
            audio = AudioSegment.from_mp3(path)
        play(audio)


if __name__ == '__main__':
    app = QApplication([])
    main = MainApp()
    main.show()
    app.exec_()
