#import matplotlib.pyplot as plt
import numpy as np
import time
#import pandas as pd
import pygame

from sklearn import svm
from sklearn.pipeline import Pipeline
#from sklearn.base import BaseEstimator, TransformerMixin

from measure import FullMeasurement
from data_process import spectro_scaler, get_calibrated_data

'''def FullMeasurement(): #Temporary function for testing
    time.sleep(1)
    spectro = np.random.uniform(0, 300, size=8)
    capa = np.random.uniform(0, 2000, size=160)
    return spectro, capa'''


file_name = "Learning_set_4.csv"

#Build the pipeline that came out the best from the data analysis in data_process.py
data, labs = get_calibrated_data(file_name, gft_only=True, drop_rec=True)
sp_scaler = spectro_scaler(param=5.555, recal=False)
clf = Pipeline(steps=[('spectro_scaler', sp_scaler), ('svm', svm.SVC(kernel='linear'))])
clf.fit(data, labs)

#Setup pygame window
pygame.init()
surface = pygame.display.set_mode((500, 500), pygame.RESIZABLE)

gft_color = (0, 255, 0)
other_color = (255, 0, 0)
cal_color = (50, 100, 200)
dark_grey = (50, 50, 50)

#Draw color of "calibration"
pygame.draw.rect(surface, cal_color, pygame.Rect(0, 0, surface.get_width(), surface.get_height()))
pygame.display.flip()

capa_empty = FullMeasurement()[1] #Calibration measurement.

#Draw color of "calibration done"
pygame.draw.rect(surface, dark_grey, pygame.Rect(0, 0, surface.get_width(), surface.get_height()))
pygame.display.flip()

stop = False

#Pygame loop
while not stop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stop = True

        #if bin_closed(): #Automatically trigger measurement by closing bin. Whether the bin is actually closed is estimated based on external light.
        #    time.sleep(0.5)
        if event.type == pygame.KEYDOWN:
            pygame.draw.rect(surface, dark_grey, pygame.Rect(0, 0, surface.get_width(), surface.get_height()))
            pygame.display.flip()

            #Perform measurement
            spectro, capa = FullMeasurement()
            capa = capa - capa_empty #Use calibration to only focus on the (rather small) differences

            #Make machine learning prediction
            new_data = np.array([np.concatenate((spectro, capa))])
            pred = clf.predict(new_data)

            #Color screen to communicate the prediction to the user
            if(pred[0] == "gft"):
                pygame.draw.rect(surface, gft_color, pygame.Rect(0, 0, surface.get_width(), surface.get_height()))
                pygame.display.flip()
            else:
                pygame.draw.rect(surface, other_color, pygame.Rect(0, 0, surface.get_width(), surface.get_height()))
                pygame.display.flip()

    #Update screen
    pygame.display.flip()
