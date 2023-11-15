from as7262 import AS7262
from grove.adc import ADC
import RPi.GPIO as GPIO
import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#Setup serial link with Arduino
ser = serial.Serial(port="/dev/ttyACM0", baudrate= 115200, timeout=1.5)
time.sleep(2)

#Constants regarding the capacitive read
num_expected_bytes = 648
packet_size = 4
startPacket = b'\x01\x01\x01\x88'
endPacket = b'\x01\x01\x01\x88'

adc = ADC(0x08) #Initialize analog input for UV and IR sensors
as7262 = AS7262() #Initialize thingy for the 6-channel spectrometer
#Setup some extra stuff for the spectrometer
as7262.set_gain(64)
as7262.set_integration_time(6.9)
#Old integration time: as7262.set_integration_time(17.857)

#Define intensity window for reference
#ref_max = np.array([1000, 10000, 10000, 10000, 10000, 10000, 10000, 100])
#ref_max = np.array([46, 378, 427, 392, 349, 350, 265, 100]) #Values from a piece of paper
#ref_min = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#ref_min = np.array([34, 170, 178, 155, 132, 102, 68, 0]) #Values of the wood

wavelengths = [940, 650, 600, 570, 550, 500, 450, 365]
mosfet = 16 #(digital) pin to control the mosfet, which is used to power the UV LED

GPIO.setmode(GPIO.BCM)
GPIO.setup(mosfet, GPIO.OUT)

def startCapaRead():
	ser.write(b"X")

def readCapacitor():
	#print("Start of read function")
	#Note: writing of X (command to start writing) is moved to a separate function for parallelization
	#ser.write(b"X")
	inbytes = ser.read(num_expected_bytes)
	#print(inbytes)
	
	#Check if start and stop match expectations
	if(inbytes[:packet_size] != startPacket or inbytes[-packet_size:] != endPacket):
		print("WARNING: Start or and packet did not match!")
	
	#print("Len inbytes:", len(inbytes))
	
	inbytes_cut = inbytes[packet_size:-packet_size]

	#Check if the inbytes are a multiple of packet_size
	if(len(inbytes_cut) / packet_size != len(inbytes_cut) // packet_size):
		print("WARNING: Inbytes not a multiple of packet_size!")
	
	N = len(inbytes_cut) // packet_size
	if(__name__ == "__main__"):
		print("N:", N)
	
	#Note: The minus 2 is because the first and last packet are removed
	#x_arr = np.zeros(num_expected_bytes // packet_size - 2)
	y_arr = np.zeros(num_expected_bytes // packet_size - 2)
	y_arr[:] = np.nan
	
	for i in range(N):
		loc = i*packet_size
		
		#Extract all bytes
		xMSB = 0
		xLSB = inbytes_cut[loc]
		yMSB = inbytes_cut[loc+1]
		yLSB = inbytes_cut[loc+2]
		zeroByte = inbytes_cut[loc+3]
		
		#The thing were zero values need to be decoded differently
		if( (zeroByte & 1) == 1):
			xLSB = 0
		if( (zeroByte & 2) == 2):
			xMSB = 0
		if( (zeroByte & 4) == 4):
			yLSB = 0
		if( (zeroByte & 8) == 8):
			yMSB = 0
		
		#Combine MSB and LSB
		#x_arr[i] = xMSB << 8 | xLSB
		y_arr[xLSB] = yMSB << 8 | yLSB
	
	if(__name__ != "__main__"):
		print("Capa read:", len(inbytes), "bytes, N=", N, "Max value:", np.max(y_arr) )
	
	return y_arr #Array with responses with fixed size


def GetSpectrumValues():
	#Turn on spectrometer
	as7262.set_measurement_mode(2)
	as7262.set_illumination_led(1)
	GPIO.output(mosfet, 1)
	
	#Read and process values
	output = np.zeros(8)
	values = as7262.get_calibrated_values()
	values_arr = list(values)
	values_arr = [x for x in values_arr]
	for i in range(len(values_arr)):
		output[i+1] = values_arr[i]
	UV_value = 0
	for a in range(50):
		UV_value += adc.read(0)
	output[7] = UV_value
	IR_value = 1000 - adc.read(3)
	output[0] = IR_value
	
	print("Raw spectro values:", output)
	
	#Remap into calibrated window
	#Linear interpolation: f(t) = A + t*(B-A) -> t = ( f(t) - A ) / (B - A)
	#output = (output - ref_min) / (ref_max - ref_min)
	
	#Turn off the spectrometer
	as7262.set_measurement_mode(3)
	as7262.set_illumination_led(0)
	GPIO.output(mosfet, 0)
	
	return output
	
def FullMeasurement(measure_time=True):
	'''Function that is responsible for a full measurement.
	This should be the only function you need from this module.'''
	
	start_time = time.time()
	startCapaRead() #Send command to Arduino to start giving capa data
	#Now first do the spectro stuff while the Arduino is still writing capa data in the serial bus
	spectro = GetSpectrumValues()
	#Now go read the capa data, which should be ready by now
	capa = readCapacitor()
	if(measure_time):
		print("Time taken for full measurement:", time.time() - start_time, "\n")
	return spectro, capa

def test_spectro():		
	spectro = GetSpectrumValues()

	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(300, 1000))
	plt.plot(wavelengths, spectro, marker='.', color='black')
	plt.xlabel("Wavelength (nm)")
	plt.ylabel("$\\frac{I - I_{min}}{I_{max}-I_{min}}$")
	ax.imshow([[0, 1], [0, 1]], cmap=plt.cm.nipy_spectral, interpolation='bicubic', extent=(400, 675, *plt.ylim()), alpha=0.5)
	ax.set_aspect('auto')
	plt.show()

def test_capa(calibrate=True):
	capa_empty = np.zeros(160)
	if(calibrate):
		startCapaRead()
		capa_empty = readCapacitor()	
	
	fig, ax = plt.subplots()
	graph = ax.plot(range(160), np.zeros(160), marker='.')[0]
	if(calibrate):
		ax.set(xlim=[0, 160], ylim=[-512, 512])
	else:
		ax.set(xlim=[0, 160], ylim=[0, 1024])
	
	def update(frame):
		startCapaRead()
		capa = readCapacitor() - capa_empty
		graph.set_ydata(capa)
		
	ani = animation.FuncAnimation(fig=fig, func=update)
	plt.show()

def test_full():
	spectro, capa = FullMeasurement()
	print("Spectro:", spectro)
	print("Capa:", capa)

if __name__ == "__main__":	
	#test_spectro()
	test_capa()
	#test_full()
