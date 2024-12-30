# Speech Recognition System
The project aims to help the speakers and the children correctly pronounce the phonemes and some words  in the Arabic language. 
  
![image](https://user-images.githubusercontent.com/82292548/209600192-53529a21-e08b-4b54-b1e9-821c98221ec9.png) 
---
![image](https://user-images.githubusercontent.com/82292548/209600217-a9a3e004-23c9-4062-a546-5293874ac3d4.png)
---
![image](https://user-images.githubusercontent.com/82292548/209600571-5200cf94-9d2f-49e5-8951-0bb3f94d65ef.png)
---
 ![image](https://user-images.githubusercontent.com/82292548/209600599-f2324299-fcaa-45b2-b50f-f7be6fd890e0.png) 
---
![image](https://user-images.githubusercontent.com/82292548/209600646-8cb3c282-26a2-45b1-9d2c-d4008ff6a52b.png)
---


# **Word Recognition System: A Comprehensive Speech Recognition Tool**

## **Project Overview**

The **Word Recognition System** is a state-of-the-art tool that utilizes **Dynamic Time Warping (DTW)**, an algorithm for measuring similarity between two temporal sequences, for recognizing spoken words in audio files. It supports different speaker categories, such as male, female, and child, by comparing audio features using **Mel Frequency Cepstral Coefficients (MFCC)**, a feature commonly used in speech processing.

This tool provides an interactive **Graphical User Interface (GUI)** built with **PyQt5**, making it easy for users to upload audio files, visualize results, and interact with the underlying algorithm. The system supports audio recognition by comparing the input sample to a set of reference samples, providing both distance metrics and classification results.

A detailed video guide explaining the **GUI** and how to use the tool is available. Please refer to the video for a complete walkthrough of the tool's functionalities.

---

## **Key Features**
- **Speaker Classification**: Classify speech into three categories: Male, Female, and Child.
- **Dynamic Time Warping (DTW)**: Use of fast DTW algorithm to compare audio features (MFCC).
- **Graphical User Interface (GUI)**: Interactive interface for easy interaction with the tool and visualizing results.
- **Testcase Management**: Organize and manage reference audio samples for each speaker category.

---

## **System Architecture**

### **1. Audio Processing Pipeline**
   The system follows these steps:
   - **Audio Preprocessing**: The input audio is preprocessed by extracting the **MFCC features** using **Librosa**. These features are used to represent the speech content in a compact form.
   - **DTW Algorithm**: The system compares the extracted features of the input sample against reference samples using the **Dynamic Time Warping (DTW)** method, which computes the "distance" between two audio signals, even if they are out of phase in time.
   - **Speaker Classification**: Based on the DTW distance, the system classifies the speaker into one of three categories: Male, Female, or Child.

### **2. Graphical User Interface (GUI)**
   The **GUI** is built using **PyQt5**, which allows users to:
   - Upload an audio file.
   - Select and compare the uploaded file with reference audio samples.
   - Visualize the results, including distance metrics, waveform plots, and MFCC features.
   - Receive classification results indicating the gender and age group of the speaker.

### **3. Speaker Reference Database**
   The system maintains a reference database of audio samples organized into three categories:
   - **Male References**: Audio samples of male speakers.
   - **Female References**: Audio samples of female speakers.
   - **Child References**: Audio samples of child speakers.
   
   Each reference database is stored in a specific directory, and users can customize the reference files.

---

## **Algorithm Workflow**

### **1. Input and Preprocessing**
   - The user uploads a test audio file (in `.wav` format) through the GUI.
   - The audio is then processed using **Librosa** to extract **MFCC** features, which represent the audio signal in a form suitable for comparison.

### **2. DTW Comparison**
   - The system calculates the **DTW distance** between the MFCC features of the input audio and those of each reference sample (Male, Female, Child).
   - The DTW distance is calculated using the **fastdtw** library, an efficient implementation of the DTW algorithm.

### **3. Speaker Classification**
   - The system computes the distances to the male, female, and child reference samples.
   - It then classifies the input audio by selecting the reference with the smallest DTW distance.
   - The corresponding gender and age group (Male, Female, Child) of the closest reference sample are returned as the classification result.

### **4. GUI Output**
   - The **PyQt5** GUI displays:
     - The test sample's waveform and MFCC plot.
     - The distance matrix comparing the test audio with reference samples.
     - The classification result (gender and age group of the speaker).

---

## **Usage**

### **Clone the Repository**

To get started with the **Word Recognition System**, clone the repository from GitHub:

```bash
git clone https://github.com/your-username/word-recognition-system.git
cd word-recognition-system/src
```

### **Install Dependencies**

Before running the project, ensure all dependencies are installed by running the following command:

```bash
pip install -r requirements.txt
```

This will install all the required libraries, including **Librosa**, **PyQt5**, **Matplotlib**, and **FastDTW**.

### **Run the Application**

Once the dependencies are installed, you can run the application by executing:

```bash
python app.py
```

This will launch the **GUI**, where you can interact with the system, upload test audio files, and view the recognition results.

### **How to Use the GUI**

1. **Upload an Audio File**: Use the "Upload" button to select an audio file (in `.wav` format) for testing.
2. **Select Reference Samples**: Choose the reference database (Male, Female, Child) for comparison.
3. **View Results**: After the comparison is complete, the tool will display:
   - **Distance Metrics**: Visualize how close the test audio is to each reference.
   - **MFCC Plot**: Visualize the MFCC features of the audio.
   - **Classification Result**: The recognized speaker's gender and age group.
4. **Customize Reference Files**: You can update the reference database by adding or modifying audio samples in the respective directories.

For a more detailed walkthrough of the tool's features and the GUI, refer to the [video guide](#) on the drive.

---

## **Algorithm Explanation**

### **Dynamic Time Warping (DTW)**
The **DTW** algorithm measures the similarity between two sequences (in this case, MFCC feature vectors of audio signals). It computes the optimal alignment between the two sequences by finding the path that minimizes the cumulative distance. The DTW distance is a measure of how similar two audio signals are, even if they are out of phase in time.

### **MFCC Features**
**MFCCs** (Mel-Frequency Cepstral Coefficients) are features that capture the short-term power spectrum of an audio signal. They are commonly used in speech recognition as they represent the audio signal in a compressed form while preserving its important characteristics. 

### **Speaker Classification**
The system classifies a given audio input based on its DTW distance to reference samples of different speaker categories:
- **Male**: Audio samples from male speakers.
- **Female**: Audio samples from female speakers.
- **Child**: Audio samples from child speakers.

The system determines the category of the input based on the minimum DTW distance to the reference samples.

---

## **Project Structure**

```
src/
│
├── app.py               # Main application file that launches the GUI
├── requirements.txt     # Python dependencies
├── reference/           # Directory containing reference audio samples
│   ├── Male/            # Male reference samples
│   ├── Female/          # Female reference samples
│   └── Child/           # Child reference samples
└── gui/                 # PyQt5 GUI components
    ├── main_window.py   # Main window and event handling
    ├── audio_player.py  # Audio playback and control
    └── visualizer.py    # Plotting MFCC and distance matrices

```

---




## **Conclusion**

The **Word Recognition System** is a powerful tool for recognizing and classifying spoken words based on audio features. It integrates **DTW** for audio comparison, **Librosa** for feature extraction, and **PyQt5** for an intuitive GUI. The system supports speaker classification and provides visual feedback to help users understand the comparison and classification process.

For more details and a complete guide on how to use the system, refer to the [video guide](https://drive.google.com/drive/folders/1-42G59hqbGSJxXrcHkwu9w4cHHOkRYvh) provided.

The [reference](https://drive.google.com/drive/folders/1-5j2qvIZct5vZj1fJrKOQ_SU7Q6AriaE) audios for the dtw. 


## **Acknowledgement**

- The project was done under the supervision **Dr/ Mohsen Rashwan** in **Cairo University Faculty of Engineering** in the department of **Electronics and Electrical Communications Engineering (EECE)**. 
- The project took the full mark. 



---

## **Contributors**
- **Abd-El-Rahman Sabry**
- **Sohila Akram**

