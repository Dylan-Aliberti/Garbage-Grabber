#import matplotlib.pyplot as plt
#import numpy as np
#import time
import pandas as pd

#Self-made modules
from measure import FullMeasurement

'''def FullMeasurement(): #Temporary function for testing
    time.sleep(1)
    spectro = np.random.uniform(0, 300, size=8)
    capa = np.random.uniform(0, 2000, size=160)
    return spectro, capa'''

num_calibrations = 10

#Prepare dictionary keys
spectral_strings = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
capacitive_strings = []
for i in range(160):
	capacitive_strings.append("C"+str(i))
#print(capacitive_strings)
data_cols = spectral_strings + capacitive_strings
#print(data_cols)
all_cols = ["date", "name", "category"] + data_cols
#print(all_cols)
	
def createDataRow(date, name, category, spectrumValues, capaValues):
	row = {"date": date, "name": name, "category": category}
	for i in range(8):
		row["S"+str(i)] = spectrumValues[i]
	for i in range(160):
		row["C"+str(i)] = capaValues[i]
	return pd.DataFrame([row])

def measure_loop(date, start_from_file, file_name):
	#Initialize data
	if(start_from_file):
		df = pd.read_csv(file_name)
		print("Succesfully loaded data.")
	else:
		df = pd.DataFrame(columns=all_cols)
		print("Starting new dataframe.")

	current_label = "calibration"
	name_counter = 0

	input("The system will first do some calibration measurements. Please press ENTER when the bin is empty and closed.")
	
	for i in range(num_calibrations):
		spectro, capa = FullMeasurement()
		name = current_label + str(name_counter) #Add automatic name: label + number. Previously this was a manual name for each object, but that cost too much time.
		name_counter += 1
		row = createDataRow(date, name, current_label, spectro, capa)
		df = pd.concat([df, row], ignore_index=True)

	print("\n\n\nCalibration done!")

	temp = "lab" #Set "temp" to "lab" so that the first label is always asked before doing any measurements.
	while True:
		#Note: temp gets set at the end of the loop. This is to make the order of the first loop better.
		if(temp == ""):
			#Perform measurement
			spectro, capa = FullMeasurement()
			name = current_label + str(name_counter) #Add automatic name: label + number. Previously this was a manual name for each object, but that cost too much time.
			name_counter += 1
			row = createDataRow(date, name, current_label, spectro, capa)
			df = pd.concat([df, row], ignore_index=True)
		elif(temp == "lab"):
			#Change current label
			while True:
				temp_label = input("Enter new label: ")
				temp_label_2 = "".join(c for c in temp_label.lower() if c.isalnum())
				if(temp_label_2 == ""):
					print("Invalid (non-alphanumeric) label. Please try again.")
				else:
					current_label = temp_label_2
					#Use this moment as an autosave moment
					df.to_csv(file_name, index=False)
					print("Autosaved!")
					break
		elif(temp == "exit"):
			#Save and quit
			df.to_csv(file_name, index=False)
			print("Saved file!")
			break
		else:
			print("Invalid input. Please try again.")

		temp = input("ENTER (leave blank): Perform measurement.\n'lab': Change current label.\n'exit': Save and quit.\n")
			
if __name__ == "__main__":

	date = input("Enter date (YYY-MM-DD): ")

	while True:
		temp = input("Start from existing file (y/n)? ")
		if(temp.lower() == "y" or temp.lower() == "yes"):
			start_from_file = True
			break
		elif(temp.lower() == "n" or temp.lower() == "no"):
			start_from_file = False
			break
		else:
			print("Invalid input. Please try again.")

	file_name = input("Enter file name you wish to use: ")

	measure_loop(date, start_from_file, file_name)
