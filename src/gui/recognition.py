import copy
import numpy as np
from glob import glob
import librosa
import scipy.spatial.distance
from fastdtw import fastdtw
from playsound import playsound
import os.path as p
from scipy.io import wavfile
from scipy.spatial.distance import euclidean, cosine
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display
from dtw import dtw
from testcase import Testcase
import pandas as pd

class WordRecognition:
    """
    A class for word recognition using Dynamic Time Warping (DTW) to compare audio samples.

    This class allows the recognition of gender and speech patterns from audio input. It compares 
    input audio samples to reference samples of male, female, and child voices. The DTW algorithm 
    is used to measure the similarity between the input and reference audio samples.

    Attributes:
        ChildReference (str): Directory path for child voice references.
        FemaleReference (str): Directory path for female voice references.
        MaleReference (str): Directory path for male voice references.
        SampleRate (int): The sample rate for audio processing (default is 16000 Hz).
    """
    ChildReference = 'Segments\\CR'
    FemaleReference = 'Segments\\FR'
    MaleReference = 'Segments\\MR'

    SampleRate = 16000

    def __init__(self):
        """
        Initializes the WordRecognition instance and sets up reference data for male, female, 
        and child voices.
        """
        self.__initialize_refs()

    def __initialize_refs(self):
        """
        Initializes the reference samples for male, female, and child voices using the 
        Testcase class for each category.
        """
        self.__ref_females = Testcase(WordRecognition.FemaleReference)
        self.__ref_children = Testcase(WordRecognition.ChildReference)
        self.__ref_males = Testcase(WordRecognition.MaleReference)

    def setReference(self, ref, type):
        """
        Sets a custom reference directory for a specific gender category.

        Args:
            ref (str): Directory path containing reference samples.
            type (str): The gender type ('M' for male, 'F' for female, 'C' for child).
        """
        if type == 'M':
            self.__ref_males = Testcase(ref)
        elif type == 'F':
            self.__ref_females = Testcase(ref)
        elif type == 'C':
            self.__ref_children = Testcase(ref)

    def getReference(self, type, i):
        """
        Retrieves a specific reference sample for a given gender and index.

        Args:
            type (str): The gender type ('M' for male, 'F' for female, 'C' for child).
            i (int): The index of the reference sample.

        Returns:
            dict: The reference sample at the given index.
        """
        if type == 'M':
            return self.__ref_males.get_case(i)
        elif type == 'F':
            return self.__ref_females.get_case(i)
        elif type == 'C':
            return self.__ref_children.get_case(i)

    @staticmethod
    def __remove_mfcc_mean(mfcc):
        """
        Removes the mean and normalizes the MFCC features for better comparison.

        Args:
            mfcc (numpy.ndarray): The MFCC features to be processed.

        Returns:
            numpy.ndarray: The processed MFCC features.
        """
        mfcc_copy = copy.deepcopy(mfcc)
        number_of_vectors = mfcc.shape[1]
        for i in range(number_of_vectors):
            mfcc_copy[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            mfcc_copy[:, i] = mfcc_copy[:, i] / np.max(np.abs(mfcc_copy[:, i]))

        return mfcc_copy

    @staticmethod
    def compare_sound(x, y):
        """
        Compares two audio samples using DTW on their MFCC features.

        Args:
            x (str or numpy.ndarray): The file path or audio data for the first sample.
            y (str or numpy.ndarray): The file path or audio data for the second sample.

        Returns:
            float: The DTW distance between the two audio samples.
        """
        # Extracting MFCC of the sound
        mfcc_1 = librosa.feature.mfcc(y=x, sr=WordRecognition.SampleRate)
        mfcc_2 = librosa.feature.mfcc(y=y, sr=WordRecognition.SampleRate)

        # Removing Mean value
        mfcc_1 = WordRecognition.__remove_mfcc_mean(mfcc_1)
        mfcc_2 = WordRecognition.__remove_mfcc_mean(mfcc_2)

        # Applying DTW
        dist, w = fastdtw(mfcc_1.T, mfcc_2.T, dist=euclidean)

        return dist

    def decide_gender(self, test):
        """
        Determines the gender of the speaker in the test audio sample.

        Args:
            test (str or Testcase): The file path or Testcase object containing the test audio.

        Returns:
            tuple: A tuple containing the predicted gender ('M', 'F', or 'C') 
                   and the corresponding reference audio sample.
        """
        if type(test) is Testcase:
            test_sample = test.get_main_sample()
            test = test_sample['wav']

        male_ref = self.__ref_males.get_main_sample()['wav']
        female_ref = self.__ref_females.get_main_sample()['wav']
        child_ref = self.__ref_children.get_main_sample()['wav']

        types = ['M', 'F', 'C']
        m_d = WordRecognition.compare_sound(test, male_ref)
        f_d = WordRecognition.compare_sound(test, female_ref)
        c_d = WordRecognition.compare_sound(test, child_ref)
        dist_list = [m_d, f_d, c_d]
        ref_list = [male_ref, female_ref, child_ref]
        min_index = dist_list.index(min(dist_list))

        return types[min_index], ref_list[min_index]

    def decide_speech(self, test, gender, index):
        """
        Determines the correct speech for the test audio sample, based on the gender.

        Args:
            test (str or Testcase): The file path or Testcase object containing the test audio.
            gender (str): The gender of the speaker ('M', 'F', or 'C').
            index (int): The index of the reference case to compare with.

        Returns:
            dict: The correct reference speech sample if a match is found.
        """
        c_test = copy.copy(test)
        if type(test) is Testcase:
            test_sample = test.get_case(index)
            c_test = test_sample['wav']

        cost_list = []

        refs = {'M': self.__ref_males, 'F': self.__ref_females, 'C': self.__ref_children}
        ref_cases = refs[gender].get_cases()
        size = len(ref_cases)
        for i in range(0, size, 2):
            ref = ref_cases[i]
            ref_wav = ref['wav']
            cost = WordRecognition.compare_sound(c_test, ref_wav)
            cost_list.append(cost)

        minimum_point = cost_list.index(min(cost_list))
        right = refs[gender].get_case(minimum_point * 2)
        return right

    def decide_speech_pair(self, test, gender, index):
        """
        Compares the test audio sample with a pair of reference speech samples to identify the correct one.

        Args:
            test (str or Testcase): The file path or Testcase object containing the test audio.
            gender (str): The gender of the speaker ('M', 'F', or 'C').
            index (int): The index of the reference case to compare with.

        Returns:
            int: 1 if the first reference is closer, 0 if the second is closer, or -1 if the match is poor.
        """
        c_test = copy.copy(test)
        if type(test) is Testcase:
            test_sample = test.get_case(index)
            c_test = test_sample['wav']

        refs = {'M': self.__ref_males, 'F': self.__ref_females, 'C': self.__ref_children}
        r_1 = refs[gender].get_case(index)['wav']
        if index == 0:
            index_2 = 1
        else:
            index_2 = index + (-1 if index % 2 else (1))
        r_2 = refs[gender].get_case(index_2)['wav']

        d_1 = WordRecognition.compare_sound(c_test, r_1)
        d_2 = WordRecognition.compare_sound(c_test, r_2)
        dists = [d_1, d_2]

        print("Indcies", index, index_2)

        right = dists.index(min(dists))
        if min(dists) > 20:
            return -1
        else:
            return ([1, 0])[right]


WordsList = [
    'ذا',
    "ثا",
    "ثا",
    "سا",
    "نذير",
    "نظير",
    "تين",
    "طين",
    "دل",
    "ضل",
    "عد",
    "عض",
    "صامد",
    "صامت",
    "ضن",
    "ظن",
    "لمس",
    "لمز",
    "مسح",
    "مسخ",
    "مصير",
    "مسير",
    "ذل",
    "زل",
    "زفر",
    "ظفر",
    "خير",
    "غير",
    "قلب",
    "كلب",
    "خائن",
    "كائن",
    "مسك",
    "مسخ",
    "علم",
    "ألم",
    "حمزة",
    "همزة",
    "حالة",
    "هالة",
    "حرم",
    "هرم",
    "حامل",
    "خامل",
    "هزم",
    "حزم",
    "الحمدلله"
]

if __name__ == '__main__':
    wr = WordRecognition()
    files = glob('Testcases/*')
    tests = []
    r = 0
    w = 0
    o = 0
    for file in files:
        t = Testcase(file + '\\Segments')
        tests.append(t)

    print('\n\n\n\n------------------------------------------------\n\n\n')

    pair        = []
    WordIndex   = []
    word        = []
    total       = {'M': [], 'F': [], 'C': []}
    other       = {'M': [], 'F': [], 'C': []}
    word_2      = {'M': [], 'F': [], 'C': []}
    word_1      = {'M': [], 'F': [], 'C': []}
    correct     = {'M': [], 'F': [], 'C': []}
    wrong       = {'M': [], 'F': [], 'C': []}

    for x in range(45):
        pair.append(int(x + 2)//2)
        currentWord = (x + 2)%2 +1
        WordIndex.append(currentWord)
        word.append(WordsList[x])
        r_g = {'M': 0, 'F': 0, 'C': 0}
        w_g = {'M': 0, 'F': 0, 'C': 0}
        o_g = {'M': 0, 'F': 0, 'C': 0}
        a_g = {'M': 0, 'F': 0, 'C': 0}
        for test in tests:
            test_main_sample = test.get_case(x)
            g = test_main_sample['speaker type']
            m = wr.decide_speech_pair(test, g, x)

            if m == 1:
                r += 1
                r_g[g] += 1
            else:
                if m == 0:
                    w += 1
                    w_g[g] += 1
                    print(f"Wrong Case [{test_main_sample['name']}  {g}  =>{test_main_sample['speaker type']}]")
                elif m == -1:
                    o += 1
                    o_g[g] += 1
                    print(f"Others Case [{test_main_sample['name']}  {g}  =>{test_main_sample['speaker type']}]")

        a = float(r) / (r + w + o)
        a_g['M'] = float(r_g['M']) / (r_g['M'] + w_g['M'] + o_g['M'])
        a_g['F'] = float(r_g['F']) / (r_g['F'] + w_g['F'] + o_g['F'])
        a_g['C'] = float(r_g['C']) / (r_g['C'] + w_g['C'] + o_g['C'])

        total['M'].append(r_g['M'] + w_g['M'] + o_g['M'])
        total['F'].append(r_g['F'] + w_g['F'] + o_g['F'])
        total['C'].append(r_g['C'] + w_g['C'] + o_g['C'])

        other['M'].append(o_g['M'])
        other['F'].append(o_g['F'])
        other['C'].append(o_g['C'])

        correct['M'].append(r_g['M'])
        correct['F'].append(r_g['F'])
        correct['C'].append(r_g['C'])

        wrong['M'].append(w_g['M'] + o_g['M'])
        wrong['F'].append(w_g['F'] + o_g['F'])
        wrong['C'].append(w_g['C'] + o_g['C'])

        if currentWord == 1:

            word_2['M'].append(w_g['M'])
            word_2['F'].append(w_g['F'])
            word_2['C'].append(w_g['C'])
            word_1['M'].append(r_g['M'])
            word_1['F'].append(r_g['F'])
            word_1['C'].append(r_g['C'])

        else:

            word_1['M'].append(w_g['M'])
            word_1['F'].append(w_g['F'])
            word_1['C'].append(w_g['C'])
            word_2['M'].append(r_g['M'])
            word_2['F'].append(r_g['F'])
            word_2['C'].append(r_g['C'])

    males = {
        'Pair' : pair,
        'pair number' : WordIndex,
        'Word' : word,
        'Males' : total['M'],
        'others' : other['M'],
        'word 2' : word_2['M'],
        'word 1' : word_1['M'],
        'correct' : correct['M'],
        'wrong' : wrong['M']
    }
    females = {
        'Pair' : pair,
        'pair number' : WordIndex,
        'Word' : word,
        'Females' : total['F'],
        'others' : other['F'],
        'word 2' : word_2['F'],
        'word 1' : word_1['F'],
        'correct' : correct['F'],
        'wrong' : wrong['F']
    }
    children = {
        'Pair' : pair,
        'pair number' : WordIndex,
        'Word' : word,
        'Children' : total['C'],
        'others' : other['C'],
        'word 2' : word_2['C'],
        'word 1' : word_1['C'],
        'correct' : correct['C'],
        'wrong' : wrong['C']
    }

    m= pd.DataFrame(males)
    f = pd.DataFrame(females)
    c = pd.DataFrame(children)

    m.to_csv('males.csv' , index=False , encoding='utf-8-sig')
    f.to_csv('females.csv', index=False, encoding='utf-8-sig')
    c.to_csv('children.csv', index=False, encoding='utf-8-sig')