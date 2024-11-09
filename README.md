# Contactless Heart Rate Monitoring Using Human Speech

## PROJECT OVERVIEW
Since, heart disease is acknowledged to be one of the main and leading causes of death worldwide, early detection is deemed to be a technique to reduce the mortality of heart rate patients. One of the most important and useful markers used to gauge one's physical state is heart rate (HR). HR information can be acquired without the need of a specific clinical gadget, which is extremely advantageous of preventative healthcare. Stroke, arrhythmia, and heart attack were amongst cardiac illnesses that can cause death, and a non-invasive gadget for monitoring heart activity can help. The human voice can be interpreted as a form of biometric data that can be used to estimate heart rate.
As a result, our goal is to create a technology that will assist doctors in detecting heart rate irregularities using only human vocal sounds.
For the detection of an anomaly in the heart rate, it is important that we extract the feature that represents the sound very well. Choosing a feature that does not accurately describe the sound can result in a poor result. Therefore, we compared and analyzed different features, MFCC and Mel-Spectrogram, for the detection of abnormalities. We've created and suggested a cutting-edge Artificial Neural Network that takes these characteristics as input and predicts if a person's heart rate is normal or abnormal.

### Project Idea
With an increase in cases of cardiac arrest, we aimed to provide aid in the health care sector for easy and quick analyzation of heart rate in detecting abnormalities in order to provide better treatment and faster treatment to the patients. Using this particular aid, the patient himself/herself can also analyze ones health.

### Motivation 
Oneiofitheimosticommonicausesiofideath is cardiac arrest. Therefore, prediction, monitoring, detection of abnormalities etc., becomes important. The existing models in the technical market lack accuracy, reliability, specificity below 90%. 
The motivation for this project is that, in the last 2-3 decades, technological advancements have boosted the rise of risk factors in human health. Cardiac arrhythmias are a significant heart disorder in which the heart signal has acquired abnormalities that cause the heartbeat to increase and decrease, resulting in rapid death. ANN approaches will aid in the early identifiaction of anomalies in arrhythmias heart beat signals, resulting in better treatment and lower medical costs. The following research question aids in the early detection of heart rate arrhythmias in humans, which allows for improved treatment.

### Objectives and Scope
#### Objectives
The project's objectives were formulated based on the purpose for the proposed work as mentioned in the previous section.

- Conduct a thorough literature review to better understand and assess the project's significance. To learn more about symptomatic sound, consult with a medical professional.
- Finding and collecting suitable datasets for training and assessing the neural network.
- Noise and undesirable frequencies are removed from the dataset by preprocessing and cleaning.
- Using a feature-based distribution of classification parameters, mathematically model a classifier for supervised training.
- Compare the observed findings with other popular and comparable algorithms and pre-processing techniques to evaluate the effectiveness of suggested algorithm in aspects of correctness, algorithmic accuracy , specificity, and sensitivity.
- To create an automated system capable of detecting abnormalities in the heart using human speech signals.
- Using various spectral properties, compare and analyse the model's performance.
- Create a scalable and efficient framework for deploying the deep learning model.
  
#### Scope
- The goal of our project is to create an AI-based algorithm that can effectively detect and forecast cardiac arrest with high specificity and sensitivity.
- The model will be able to speed and simplify difficult data processing and improve diagnosis after training and testing using a balanced dataset.
- This will help clinics reduce wait times and enhance workflow. In a time and budget-constrained environment, this model will aid in altering patient care benchmarks.
  
#### Hardware and Software Requirements
##### Hardware Requirements
Hardware: LM393 Sound Detection Sensor Module for Node MCU detects whether the sound has exceeded a threshold value. The sound is fed into an LM393 op-amp by detecting the sound using a microphone. An onboard potentiometer is used to set the setpoint of the sound level. 

##### Software Requirements
-	Optical Heart Sensor (Health App)
-	Librosa is a Python package that allows you to analyse audio signals.
-	The Jupyter Notebook acts as a free, open source application. It lets the data scientists create and share documents that include equations, computational output, live code, visualisations and other multimedia elements.
-	TensorFlow - Machine intelligence library. 
-	Keras - Python interface to TensorFlow.

## System Architecture
![image](https://github.com/user-attachments/assets/5e04727d-ab7c-44eb-9523-4a99848d3e66)

## Layered Architecture
![image](https://github.com/user-attachments/assets/f35f79fe-7c52-4cfe-ae07-bbb22f6395c6)

## Data Collection
Data was gathered from 104 different subjects aged between 20-69. Speech and Heart rate were recorded in a closed room with minimum disturbance for better sound frequency, also 6 different emotions were captured such as anger, joy, sadness and also post exercise variations. Data was collected in form of maximum 7 second recording for feature extraction. Voice was recorded on LM393 sound sensor and the hear rate was detected simultaneously on an apple smart I watch series 4. Photoplethysmography is used by the optical heart sensor of the Apple watch. The entire data collection process was completed in 45 days. 

## DATA PREPROCESSING
The dataset that collected was saved in a folder path. The first step was to extract this dataset into the code from this path, and iteration is used to extract each row of the dataset. 
Librosa is a library in python that is used for the analysis of audio signals. Librosa consists of several modules which can be used to achieve this purpose. 

Feature extraction is performed for finding traits from sounds. In our program, we have created a function feature_extraction, passed the audio files to this function and have carried out the process the above preprocessing steps. 

- Preemphasis: Preemphasis is the first step of feature extraction. Preemphasis imporoves the signal to noise ratio. This means that it increases the actual sound that was meant to be captured and reduces the unnecessary noise that is insignificant for our model. The audio signal was loaded with a sampling rate of 44100, which means 44100 samples per second, as this is the standard used for most audio signals. This signal is filtered with a filter with a coefficient of 0.95.
The next step is to compute a mel scaled spectrogram using the melspectogram function of librosa. The preemphasized signal is passed as the audio time series with sampling rate of 44100. The number of bands is n_mels=42. 

- Framing: The procedure of partitioning human voice into smaller frames with 50% overlap between consecutive frames is regarded as framing. The hop length of the frames is kept as 100 and window length is 256. This means that there are 100 samples between two frames.

- Windowing: Windowing is a process to remove any discontinuities in the audio signal by bending it and altering the signals to slightly zero at beginning and the end. We have used hanning window function in our program as this function makes the signals touch zero at both ends; as compared to hamming window, which stops the signal slightly above zero thus not touching zero, which leaves slight discontinuities. 
 
- Fast Fourier Transform (FFT): FFT is the shortest step to generate Discrete Fast Fourier Transform (DFT). The initial signals are transformed into discrete frequency domain components utilising FFT. The length of FFT window has been kept as 512 (n_fft=512), which is the recommended value. The window length is padded with zeros to match the length of FFT window. n_fft is always greater than or equal to window length.

- Mel Filter Bank Processing: In mel filter bank processing, the vocal signals do not follow a linear scale. As a response, the mel scale must be used to filter. The mel filter bank, which contains of overlapping triangular filters, is then used to derive the weighted sum of filter components. This converts the output to ai= meli= scale estimation. Mel filters having base frequencies that have been linearly scattered and a uniform band upon this mel scale.
  
- Discrete Cosine Transform: This step is used to invert the mel spectrum derived in the prior step in the form of a log into the time domain, resulting the Meli= Frequency Cepstrum Coefficient (MFCC). The mel spectrogram generated is then passed to the mfcc function in our program to get 12 mel frequency cepstrum coefficients, which are returned. Discrete Cosinne Transform (DCT) type is kept as 2, which is the default type. We then take the mean of the transposed values in order to feed them as an array to predict the audio labels.

## TRAINING AND TESTING
After feature extraction is complete, we have appended the extracted features in the form of array. The labels formed are data of voice recordings, gender, age group, emotional state, heart rate of each sample of the dataset. We have taken recordings of 2 genders namely Male (M) and Female (F). We have included 6 different states and 5 different age groups. The descriptor "state" is used to describe the subject's emotional or physical state. After this, we performed one-hot encoding on gender, state and age group, and then appended to the metadata as an array. In the next step, we converted the array into list form. We finally achieve a dataset that consists of one-hot encoded data and we save this dataset in csv format. This csv file is used in the next steps to perform training, testing and other operations.
We have used the TensorFlow library for further steps and training and testing of the dataset. TensorFlow is a Python library which is open source. It is one of the most used library in Deep Learning because it particularly supports training and evaluation of neural networks. Other libraries most commonly used other than TesorFlow are PyTorch and Keras libraries. Moreover, TesorFlow helps in building various high level models as compared to PyTorch library.
The dataset is subsequently ordered through at training with split=0.8, which tends to mean that the training set constitutes 80% of the total samples and the testing set includes 20%. The training set is taken higher than testing set because the model would be better trained in a larger dataset. For training the model, the batch size is taken to be 5, which means that 6 training examples would be used in each iteration. The epochs of 4 is taken.

## ANN STRUCTURE
We have used Artificial Neural Network (ANN) for classification. ANN is a set of layers that are interconnected to each other and comprises of artificial neurons in three layers. The input features are there in the first layer, which is bonded to the output layer via differing intervening hidden layers that are interconnected. The input variables are processed by each neuron and the values computed are passed to the neuron present in the subsequent layer. The ANN loosely represents the neurons of a biological brain. We have used the Sequential model in which each layer consists of exactly one input tensor and one output tensor. In our model, we have added the first layer, three intermediate layers and the final output layer for the neural network. There's 200 neurons in the first layer and one neuron in the output layer. The activation function we will be using for our Artificial Neural layers is ReLU. We have a small Dropout value of 20% on our ANN layers. An activation function is used to define how the set of inputs would be transformed to the output neuron.

## Results
We've devised a technique that uses speech signals to assist doctors in accurately detecting heart rate. The system extracts features from an audio file and feeds them into the neural network as an input. On the human speech audios, we applied Artificial Neural Networks to achieve good results for any research done in this field using human speech and audio signals. 
Overall, we derived an accuracy of 0.746 ᵙ 0.75.
From the performance analysis, we obtained the precision as follows: Micro precision: 0.75, Macro precision: 0.65 and Weighted Precision: 0.73. 
We also received the Recall values as followed: Micro Recall: 0.75, Macro Recall: 0.62 and Weighted Recall: 0.75.
The F1-scores of the prediction stand as follows: Micro F1-score: 0.75, Macro F1-score: 0.63 and finally the weighted Fi-score of 0.73, 
and as conducted and seen through our research, we have observed this to be the highest accuracy predicted in the field of heart rate.

## CONCLUSION AND FUTURE SCOPE
We predicted the heart rate after training the model in this study with a labelled dataset and pre-identified heart rate in order to classify into normal and abnormal heart rates. The Dataset in our particular research has been created with one on one interaction with the subjects and covers various important factor in order to detect heart rate. This study also overviews various algorithms and functions which are needed to pre-process the data in order to obtain optimum results. We've devised a technique that uses speech signals to assist doctors in accurately detecting heart rate. The system extracts features from an audio file and feeds them into the neural network as an input. On the human speech audios, we applied Artificial Neural Networks to achieve good results for any research done in this field using human speech and audio signals. We acquired a system 75% accuracy.
We also created a varied dataset which contained essential features like age, which is helpful while deploying a model. The snag we ran across have been that the dataset was not sufficiently large. If a huge dataset is available, deep learning and machine learning results can improve dramatically. When compared to other researchers, we found that the algorithm we used in ANN architecture boosted accuracy. When the dataset size is expanded, deep learning with many other optimizations can be employed to produce more promising outcomes. Machine learning and a variety of other optimization approaches can also be applied to improve the evaluation findings. The data can be normalised in a variety of ways, and the results can be compared. More techniques to integrate heart-disease-trained ML and DL models with specific multimedia for the convenience of patients and clinicians could be discovered.










