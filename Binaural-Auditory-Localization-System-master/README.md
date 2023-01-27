# Binaural-Auditory-Localization-System

Find the angle where the sound comes.

Requirements:
1. Brian v1.4 (may need to degrade your numpy and scipy)
2. Cython
3. Nengo

To run:

1. SoundSplitter:
Put the wav file into SoundSplitter folder and run SoundSplitter.py .It will produce two new wav files.
Create two new directories 'wav' and 'txt' under Cython_IPEM folder and move the 2 new wav files from SoundSplitter to the 'wav' folder

2. Cython_IPEM:
Run setup.py in Cython_IPEM to setup IPEM and then run IPEM.py. A txt file is produced in 'txt' folder. Change the channel at line 91 in IPEM.py (L to R or R to L) and run again. Another txt file should also be produced in 'txt' folder.
Move the two txt files to LSO_MODEL and MSO_MODEL folders. 

3. LSO_MODEL, MSO_MODEL:
Run LSO_Model.py and MSO_Model.py in LSO_MODEL and MSO_MODEL folders respectively. Both will produce 40 txt files representing a 40-channel frequency band. This may take you some time...
Create two new directiries 'LSO' and 'MSO' under IC_MODEL folder. Move the two 40 txt files to 'LSO' and 'MSO' respectively.

4. IC_MODEL:
Run IC_Model.py and it will porduce 40 new txt files. 
Create a new directory, 'IC' under Angle_Estimation folder and Move the 40 new txt files to 'IC'.

5. Angle_Estimation:
Run Angle_Estimation.py and the result is stored in the 'out' array, each column representing the time steps and each row the divided angle from -90 to 90 degrees.(In this case, the 7 rows are -90,-60,-30,0,30,60,90). 
