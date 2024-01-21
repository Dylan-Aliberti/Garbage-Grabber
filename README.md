# Garbage-Grabber
This repository contains all code from the Garbage Grabber project. The garbage grabber is a prototype for a smart garbage bin that can automatically separate waste. This bin uses a spectrograph and a capacitive sensor, coupled with machine learning, to determine what category of waste is in it. This project was done by Dylan Aliberti, Brammert Habing, Tim Mulder, and Jonathan Pilgram, as an internship at Cyclus N.V. For more information, please refer to our report in the link below.

Within the Garbage Grabber, the python files can be run from the Raspberry Pi. Here is a quick overview of the four files:
- measure: A module that interacts with the Grove sensors and provides functions that manage the whole measuring process. This is meant to import in the other files, but can also be executed on its own to test sensors.
- datacollect: This program guides the user through a measurement loop to measure and collect data into a Pandas dataframe for later use.
- data_process: A program containing lots of functions for analysing the obtained dataset and tuning hyperparameters for machine learning. The purpsose of this file is to determine the best processing pipeline and machine learning method/parameters for identifying waste.
- identify: Run this part to make the bin actually identify waste. The screen turns green for organic waste, and red otherwise.

The capacitive measurement is not a simple Grove module, but a whole Circuit on its own, driven by an Arduino. The capacitive measurement was based on the following Instructables: https://www.instructables.com/Touche-for-Arduino-Advanced-touch-sensing/ . The Arduino code in the Garbage Grabber repository is a modified version of the corresponding code from the Instructable: https://github.com/Illutron/AdvancedTouchSensing . In the Garbage Grabber, the Arduino was connected to the Raspberry Pi via USB, so that the capacitive data can be further processed in the Python files mentioned above.

The report can be found in the main folder.
