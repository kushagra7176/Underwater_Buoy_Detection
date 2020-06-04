
from EM_GMM_Algorithm import *
from Final_Buoy_Detection import *
from os import path
import csv



def train():
    Gaussian_List = []
    mean_yellow_1, mean_yellow_2, mean_yellow_3, std_yellow_1, std_yellow_2, std_yellow_3 = EM_GMM_algorithm("DATASET/Yellow/",
                                                                                                   "Yellow")
    mean_orange_1, mean_orange_2, mean_orange_3, std_orange_1, std_orange_2, std_orange_3 = EM_GMM_algorithm("DATASET/Orange/",
                                                                                                   "Orange")
    mean_green_1, mean_green_2, mean_green_3, std_green_1, std_green_2, std_green_3 = EM_GMM_algorithm("DATASET/Green/", "Green")

    gaussian_yellow = compute_Gaussian_Equations(mean_yellow_1, mean_yellow_2, mean_yellow_3,
                                                 std_yellow_1, std_yellow_2, std_yellow_3, 'Yellow')

    Gaussian_List.append(gaussian_yellow)

    gaussian_orange = compute_Gaussian_Equations(mean_orange_1, mean_orange_2, mean_orange_3,
                                                 std_orange_1, std_orange_2, std_orange_3, 'Orange')
    Gaussian_List.append(gaussian_orange)

    gaussian_green = compute_Gaussian_Equations(mean_green_1, mean_green_2, mean_green_3,
                                                std_green_1, std_green_2, std_green_3, 'Green')
    Gaussian_List.append(gaussian_green)

    return Gaussian_List

def load_model_parameters():
    Gaussian_List = []
    with open('train_data_Yellow.csv', 'r') as file1:
        reader = csv.reader(file1)
        rows_yellow = [r for r in reader]
        data_yellow = rows_yellow[50]
        gaussian_yellow = compute_Gaussian_Equations(float(data_yellow[1]), float(data_yellow[2]), float(data_yellow[3]),
                                                    float(data_yellow[4]), float(data_yellow[5]), float(data_yellow[6]), 'Yellow')
        Gaussian_List.append(gaussian_yellow)

    with open('train_data_Orange.csv', 'r') as file2:
        reader = csv.reader(file2)
        rows_orange = [r for r in reader]
        data_orange = rows_orange[50]
        gaussian_orange = compute_Gaussian_Equations(float(data_orange[1]), float(data_orange[2]), float(data_orange[3]),
                                                    float(data_orange[4]), float(data_orange[5]), float(data_orange[6]), 'Orange')
        Gaussian_List.append(gaussian_orange)


    with open('train_data_Green.csv', 'r') as file3:
        reader = csv.reader(file3)
        row_green = [r for r in reader]
        data_green = row_green[50]
        gaussian_green = compute_Gaussian_Equations(float(data_green[1]), float(data_green[2]), float(data_green[3]),
                                                    float(data_green[4]), float(data_green[5]), float(data_green[6]), 'Green')
        Gaussian_List.append(gaussian_green)

    return Gaussian_List



########################################################################################################################
######################################                    MAIN                    ######################################
########################################################################################################################

# Check if Pre-Trained Model parameters are available
if path.exists("train_data_Yellow.csv") and path.exists("train_data_Orange.csv") and path.exists("train_data_Green.csv") :
    print("Pre-Trained Data Exists. Reading Gaussian parameters from trained data... ")
    Gaussian_list = load_model_parameters()
else:
    Gaussian_list = train()

# Perform Buoy Detection
buoy_Detection(Gaussian_list)