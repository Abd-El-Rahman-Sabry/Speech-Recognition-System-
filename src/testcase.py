import glob
import os
import re
import librosa
from pydub import AudioSegment


class Testcase:
    MustAll = True

    def __init__(self, name):
        self.__name = name
        self.__testcases = []
        if not self.__wav_exists():
            self.__generate_wav()
        else:
            self.__read_exists_data()

    def get_cases(self):
        return self.__testcases

    def get_case(self , i):
        return self.__testcases[i]

    def get_main_sample(self):
        return self.__testcases[46]

    @staticmethod
    def __extract_information(x):
        # x = 'G02S1F22MP01W1R'
        ex = re.compile(r'G(\d+)S(\d)(\S)(\d+)[WM]P(\d+)W?(\d)?')
        a = re.findall(ex, x)
        a = a[0]
        sample = {}
        sample['name'] = x
        if len(a):
            sample['student_number'] = int(a[1])
            sample['speaker type'] = a[2]
            sample['speaker age'] = int(a[3])
            sample['word pair'] = int(a[4])
            sample['word'] = int(a[5]) if a[5] != '' else 1
        else:
            sample['student_number'] = 0
            sample['speaker type'] = 'None'
            sample['speaker age'] = 0
            sample['word pair'] = 0
            sample['word'] = 0

        return sample

    # Generate Wav files if it's mp3
    def __wav_exists(self):
        if os.path.exists(self.__name + '\\Wav'):
            if Testcase.MustAll:
                files = glob.glob(self.__name + '\\Wav\\*.wav')
                if len(files) == 47:
                    return True
                else:
                    return False
            else:
                return True

        else:
            return False

    def __generate_wav(self):
        sound_files = glob.glob(self.__name + '\\*.mp3')
        os.mkdir(self.__name + '\\Wav')
        current_path = os.getcwd()
        os.chdir(self.__name + '\\Wav')
        for file in sound_files:
            old_name = os.path.basename(os.path.normpath(file))
            new_name = (old_name).split('.')[0] + '.wav'
            sound = AudioSegment.from_mp3('..\\' + old_name)
            sound.export(new_name, format='wav')
            sample = Testcase.__extract_information(new_name)
            t, f = librosa.load(new_name, sr=None)
            sample['wav'] = t
            self.__testcases.append(sample)

        os.chdir(current_path)

    def __read_exists_data(self):
        files = glob.glob(self.__name + '\\Wav\\*.wav')

        for file in files:
            name = os.path.basename(os.path.normpath(file))
            print(name)
            sample = self.__extract_information(name)
            t, f = librosa.load(file, sr=None)
            sample['wav'] = t
            self.__testcases.append(sample)
            print(sample)


if __name__ == '__main__':
    test = Testcase('Segments\\CR')
