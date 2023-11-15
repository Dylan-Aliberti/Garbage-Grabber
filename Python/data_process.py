'''Module for testing different analysis and machine learning methods.'''
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
#from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

file_name = "Learning_set_4.csv"
#file_name = "GGf_capatest.csv"
#file_name = "test3.csv"

#The wavelengths of the spectrometer
wavelengths = [940, 650, 600, 570, 550, 500, 450, 365]
#spectrometer unit: counts/microWatt/cm^2. Measurement was done with gain 64 and integration time 6.9
capa_mv_per_unit = 4.9 #5000 millivolts divided over 1024 steps

freqs = np.array(list(range(160)))
#Arduino system clock: 16 MHz. Prescaler used: 1. The array represenets all the compare match registers.
#Frequency = (system clock speed) / [ (prescaler) * (compare match register + 1) ]
freqs = 16e6 / (freqs + 1)
#Plotting capa vs frequency will not give nice readable results though. Use period instead.
periods = 1 / freqs #1/Hz = s
periods *= 1e6 #seconds -> microseconds

color_dict = {
    "gft": "green",
    "glass": "yellow",
    "metal": "gray",
    "opk": "blue",
    "pmd": "red",
    "res": "purple",
    "other": "red"
}


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




#---------------------------------------------------------------------
# Data pre-processing
#---------------------------------------------------------------------

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def extract_df(df, drop_rec=True):
    if(drop_rec):
        #Drop recycling center entries
        df = df.drop(df[df['category'] == 'rec'].index)
    
    #Split calibration rows from rest of the rows
    cal_df = df.loc[df['category'] == "calibration"]
    rest_df = df.loc[df['category'] != "calibration"]

    #Extract capacitive and spectral data from each row, separately for calibration and for actual objects.
    cal_capa = cal_df[capacitive_strings].to_numpy()
    cal_spectro = cal_df[spectral_strings].to_numpy()
    capa = rest_df[capacitive_strings].to_numpy()
    spectro = rest_df[spectral_strings].to_numpy()

    labs = rest_df["category"].to_numpy()

    return cal_capa, cal_spectro, capa, spectro, labs

def get_calibrated_data(file_name, gft_only=True, drop_rec=True):
    '''Function that reads and extracts data from filename using "extract_df".
    The capacitive reads are automatically calibrated based on the data in category "calibration".
    Returns: Tuple (data, labs). data is an Lx168 numpy array containing spectro and capa data concatenated. Labs is an size L array containing labels.'''

    df = pd.read_csv(file_name)

    cal_capa, cal_spectro, capa, spectro, labs = extract_df(df, drop_rec)

    #Substract empty measurement from capacitive reads
    capa_empty = np.mean(cal_capa, axis=0)
    for i in range(len(capa)):
        capa[i] = capa[i] - capa_empty

    #Convert capa values back to real (Delta) voltages
    capa *= capa_mv_per_unit
    #for i in range(len(capa)):
    #    capa[i] = capa[i] * capa_mv_per_unit

    #Substract empty measurement from spectro reads
    #spectro_empty = np.mean(cal_spectro, axis=0)
    #for i in range(len(spectro)):
    #    spectro[i] = spectro[i] - spectro_empty

    #Concatenate data
    data = np.concatenate((spectro, capa), axis=1)

    #Test only GFT vs other
    if(gft_only):
        for i in range(len(labs)):
            if(labs[i] != "gft"):
                labs[i] = "other"

    return data, labs

def reweigh_spectro_vs_capa(data):
    data[:,:8] = data[:,:8] * 0.01

class spectro_scaler(BaseEstimator, TransformerMixin):
    def __init__(self, param=0.25, recal=True):
        self.param = param
        self.recal = recal

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        temp = X_[:,:8]

        if(self.recal): #Recalibrate values so that range of features is always from 0 to 1.
            for i in range(8):
                value_min = np.min(temp[:,i])
                value_max = np.max(temp[:,i])
                #print(value_min, value_max)
                #Linear resizing. f(t) = A + t * (B - A) --> t = ( f(t) - A ) / (B - A)
                temp[:,i] = ( temp[:,i] - value_min ) / (value_max - value_min)

        temp = temp * self.param #Common rescale of spectral data
        X_[:,:8] = temp
        return X_

#--------------------------------------------------------------------------------
# Dimension reduction
#--------------------------------------------------------------------------------

def spectro_moments(spectro_arr, num_moments=4):
    moments_arr = np.zeros((len(spectro_arr), num_moments))
    for i in range(len(moments_arr)):
        c = np.average(wavelengths, weights=spectro_arr[i]) #C for central, which is the weighted mean
        moments_arr[i][0] = c
        moments_arr[i][1] = np.average( (wavelengths - c)**2, weights=spectro_arr[i]) #Variance
        sigma = np.sqrt(moments_arr[i][1]) #Standard deviation
        for j in range(2, num_moments):
            power = j+1
            moments_arr[i][j] = np.average( (wavelengths - c)**power, weights=spectro_arr[i]) / sigma**power
    return moments_arr

def capa_STFT_means(capa_arr, num_windows, excl_dc=False):
    '''Reduces the number of dimensions for capacitive reading using Short-Term Fourier Transformation and taking the weighted mean.
    Input and output are 2D numpy arrays. Note that for machine learning, the labels have to be provided outside of this function.'''
    N = len(capa_arr[0]) #Number of readings in graph
    num_rows = len(capa_arr)
    window_size = N // num_windows
    N_fft = window_size // 2 + 1
    readnumbers = np.linspace(0, N_fft-1, N_fft)
    if(excl_dc):
        readnumbers = readnumbers[1:] #
    #print(readnumbers)
    averages = np.zeros((num_rows, num_windows))
    for row in range(num_rows):
        for i in range(num_windows):
            arr = capa_arr[row]
            temp_fft = np.absolute(np.fft.rfft(arr[i*window_size:(i+1)*window_size]))
            if(excl_dc):
                temp_fft = temp_fft[1:] #Filter out the DC component
            averages[row][i] = np.average(readnumbers, weights=temp_fft)
    return averages


#--------------------------------------------------------------------------------
# Data visualisation
#--------------------------------------------------------------------------------
def data_multi_plot(data_arr, labs, wanted_labels):
    plt.figure()
    for i in range(len(data_arr)):
        if(labs[i] not in wanted_labels):
            plt.plot(data_arr[i], '.', color="red")
    for i in range(len(data_arr)):
        if(labs[i] in wanted_labels):
            plt.plot(data_arr[i], '.', color="green")
    plt.show()

def print_dat_labeled(dat, labs):
    for i in range(len(dat)):
        print(dat[i], labs[i])

def plot_capa(data, labs, index):
    '''Function that plots the capacitive data of a specific index inside the data.'''
    capa = data[index, 8:]
    plt.figure()
    plt.plot(periods, capa, marker='.')
    plt.title("Category: " + str(labs[index]))
    plt.xlabel("Period (μs)")
    plt.ylabel("Response (mV)")
    plt.show()

def plot_spectro(data, labs, index):
    '''Function that plots the spectral data of a specific index inside the data.'''
    #Recalibrate
    spectro = data[:,:8]
    sp_scaler = spectro_scaler(param=1, recal=True)
    spectro = sp_scaler.transform(spectro)

    #Select the wanted index out of all the data
    spectro = spectro[index, :8]

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(300, 1000))
    plt.plot(wavelengths, spectro, marker='.', color='black')
    plt.title("Category: " + str(labs[index]))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("$\\frac{I - I_{min}}{I_{max}-I_{min}}$ (feature-wise rescaled)")
    ax.imshow([[0, 1], [0, 1]], cmap=plt.cm.nipy_spectral, interpolation='bicubic', extent=(400, 675, *plt.ylim()), alpha=0.5)
    ax.set_aspect('auto')
    plt.show()

def scroll_capa_plots(data, labs, jump=1):
    capa = data[:,8:]
    for i in range(len(capa)//jump):
        plt.figure()
        for j in range(jump):
            index = i*jump+j
            plt.plot(capa[index])
        plt.title("i="+str(i)+": "+str(labs[index]))
        plt.show()

def multiplot(data, labs, indices, presmode=True):
    #Capa
    capa = data[:, 8:]
    plt.figure()
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    if(presmode):
        plt.rc('font', size=14)    
    for i in indices:
        if(labs[i] == "gft"):
            label="Organic"
        elif(labs[i] == "res"):
            label="Residual"
        else:
            label=labs[i]
        #plt.plot(periods, capa[i], marker='.', label=label)
        if(label == "Organic"):
            plt.plot(periods, capa[i], label=label, color="C2")
        else:
            plt.plot(periods, capa[i], label=label)
    #plt.title("Category: " + str(labs[indices[0]]))
    plt.xlabel("Period (μs)")
    plt.ylabel("Response (mV)")
    plt.legend()
    plt.show()

    #Spectro
    '''spectro = data[:,:8]
    sp_scaler = spectro_scaler(param=1, recal=True)
    spectro = sp_scaler.transform(spectro)

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(300, 1000))
    if(len(indices) == 2):
        colors = {indices[0]: "black", indices[1]: "blue"}
        for i in indices:
            plt.plot(wavelengths, spectro[i], marker='.', color=colors[i], label=labs[i])
    else:
        for i in indices:
            plt.plot(wavelengths, spectro[i], marker='.', color='black')
    plt.title("Category: " + str(labs[indices[0]]))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("$\\frac{I - I_{min}}{I_{max}-I_{min}}$ (feature-wise rescaled)")
    ax.imshow([[0, 1], [0, 1]], cmap=plt.cm.nipy_spectral, interpolation='bicubic', extent=(400, 675, *plt.ylim()), alpha=0.5)
    ax.set_aspect('auto')
    plt.legend()
    plt.show()'''


def spectro_capa_scatter_plot(data, labs, target="gft"):
    '''This function makes a scatter plot of total capacitive response vs total spectral response.'''
    L = len(data)
    spectro_scalars = np.zeros(L)
    capa_scalars = np.zeros(L)

    spectro = data[:,:8]
    sp_scaler = spectro_scaler(param=1, recal=True)
    spectro = sp_scaler.transform(spectro)

    for i in range(L):
        spectro_scalars[i] = np.mean(np.absolute(spectro[i]))
        #spectro_scalars[i] = np.sum(data[i][:8])
        capa_scalars[i] = np.mean(np.absolute(data[i][8:]))

    unique_classes, counts = np.unique(labs, return_counts=True)

    sp = np.empty(len(unique_classes), dtype=object)
    ca = np.empty(len(unique_classes), dtype=object)
    #sp = np.zeros((len(unique_classes), 0))
    #ca = np.zeros((len(unique_classes), 0))
    for i in range(len(unique_classes)):
        sp[i] = np.zeros(0)
        ca[i] = np.zeros(0)
        for j in range(L):
            if(labs[j] == unique_classes[i]):
                sp[i] = np.append(sp[i], spectro_scalars[j])
                ca[i] = np.append(ca[i], capa_scalars[j])

    plt.figure()
    for i in range(len(unique_classes)):
        plt.loglog(sp[i], ca[i], 'o', label=unique_classes[i], color=color_dict[unique_classes[i]])
    plt.xlabel("Log-Rescaled absolute spectral difference (-)")
    plt.ylabel("Mean absolute capacitive difference (log mV)")
    plt.legend()
    plt.show()

def print_confusion_results(conf_mat, labs):
    '''Function that prints the class-specific results of the GFT vs non-GFT classification problem.'''
    unique_classes, counts = np.unique(labs, return_counts=True)
    print(unique_classes)
    print(counts)
    gft_index = np.where(unique_classes == "gft")[0][0]

    num_gft = counts[gft_index]
    num_nongft = np.sum(counts) - num_gft

    gft_true_positives = conf_mat[gft_index][gft_index]
    gft_false_positives = np.sum(conf_mat[:,gft_index]) - gft_true_positives
    gft_false_negatives = np.sum(conf_mat[gft_index, :]) - gft_true_positives
    mat_masked = np.ma.array(conf_mat, mask=False)
    mat_masked.mask[gft_index,:] = True
    mat_masked.mask[:,gft_index] = True
    gft_true_negatives = np.sum(mat_masked, axis=(0, 1))

    gft_TP_perc = round( gft_true_positives / num_gft * 100 )
    gft_FP_perc = round( gft_false_positives / num_nongft * 100 )
    gft_TN_perc = round( gft_true_negatives / num_nongft * 100 )
    gft_FN_perc = round( gft_false_negatives / num_gft * 100 )

    print("GFT true positives: " + str(gft_true_positives) +  " (" + str(gft_TP_perc) + r"% of GFT)")
    print("GFT false positives: " + str(gft_false_positives) +  " (" + str(gft_FP_perc) + r"% of non-GFT)")
    print("GFT true negatives: " + str(gft_true_negatives) +  " (" + str(gft_TN_perc) + r"% of non-GFT)")
    print("GFT false negatives: " + str(gft_false_negatives) +  " (" + str(gft_FN_perc) + r"% of GFT)")

    for i in range(len(counts)):
        print(str(unique_classes[i]) + ": " + str(conf_mat[i][i]) + "/" + str(counts[i]) + " (" + str(round(conf_mat[i][i] / counts[i] * 100)) + "%)")

def confusion_matrices(clf, data, labs, folds):
    result = np.zeros((folds, 2, 2))
    kf = KFold(folds)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labs[train_index], labs[test_index]

        clf.fit(X_train, y_train)
        result[i] = confusion_matrix(y_test, clf.predict(X_test))
    return result

if __name__ == "__main__":
    data, labs = get_calibrated_data(file_name, gft_only=True, drop_rec=True)
    #
    #plot_capa(data, labs, 0)
    #plot_spectro(data, labs, 0)

    #multiplot(data, labs, [4, 34, 57], presmode=True)

    '''for i in range(32, len(data)):
        print(i)
        plot_spectro(data, labs, i)
        plot_capa(data, labs, i)'''

    sp_scaler = spectro_scaler()
    clf = Pipeline(steps=[('spectro_scaler', sp_scaler), ('svm', svm.SVC(kernel='linear'))])

    param_grid = {
    "spectro_scaler__param": np.linspace(2, 10, 10),
    "spectro_scaler__recal": [True, False],
    "svm__kernel": ['linear', 'rbf']
    }

    folds = 10

    search = GridSearchCV(clf, param_grid, cv=folds, scoring='accuracy')
    search.fit(data, labs)

    clf_best = search.best_estimator_
    
    print("Best parameters:", search.best_params_)
    print("Best accuracy:", search.best_score_)

    print(len(data))

    skf = StratifiedKFold(n_splits=folds)
    #scores = cross_val_score(clf_best, data, labs, cv=skf, scoring='accuracy')
    cv_results = cross_validate(clf_best, data, labs, cv=skf)
    preds = cross_val_predict(clf_best, data, labs, cv=skf)
    print("Test scores:", cv_results['test_score'])
    print("Average:", np.mean(cv_results['test_score']), ". Std:", cv_results['test_score'].std())

    conf_mat = confusion_matrix(labs, preds)
    print(conf_mat)
    print_confusion_results(conf_mat, labs)

    spectro_capa_scatter_plot(data, labs)
