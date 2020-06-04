
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import csv
import random

def calculateGaussianEquation(x_co, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-np.power(x_co - mean, 2.) / (2 * np.power(std, 2.)))


def calculateProbabilty(x_co, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * (math.exp(-((x_co - mean) ** 2) / (2 * (std) ** 2)))

def expectation_step(pixel1, pixel2, mean1, mean2, mean3, std1=10, std2=10, std3=10):
    B_px1_b = []
    B_px1_g = []
    B_px1_r = []

    B_px2_b = []
    B_px2_g = []
    B_px2_r = []

    B = []

    p_b = 1/3
    p_g = 1/3
    p_r = 1/3

    # Iterate over Green channel pixels
    for px1 in pixel1:
        P_px1_b = calculateProbabilty(px1, mean1, std1)
        P_px1_g = calculateProbabilty(px1, mean2, std2)
        P_px1_r = calculateProbabilty(px1, mean3, std3)

        # Compute the Bayesian Probabilities.
        B_px1_b.append((P_px1_b * p_b) / ((P_px1_b * p_b) + (P_px1_g * p_g) + (P_px1_r * p_r)))
        B_px1_g.append((P_px1_g * p_g) / ((P_px1_b * p_b) + (P_px1_g * p_g) + (P_px1_r * p_r)))
        B_px1_r.append((P_px1_r * p_r) / ((P_px1_b * p_b) + (P_px1_g * p_g) + (P_px1_r * p_r)))

    # Iterate over Red channel pixels
    for px2 in pixel2:
        P_px2_b = calculateProbabilty(px2, mean1, std1)
        P_px2_g = calculateProbabilty(px2, mean2, std2)
        P_px2_r = calculateProbabilty(px2, mean3, std3)

        # Compute the Bayesian Probabilities.
        B_px2_b.append((P_px2_b * p_b) / ((P_px2_b * p_b) + (P_px2_g * p_g) + (P_px2_r * p_r)))
        B_px2_g.append((P_px2_g * p_g) / ((P_px2_b * p_b) + (P_px2_g * p_g) + (P_px2_r * p_r)))
        B_px2_r.append((P_px2_r * p_r) / ((P_px2_b * p_b) + (P_px2_g * p_g) + (P_px2_r * p_r)))

    B.append([B_px1_b, B_px1_g, B_px1_r , B_px2_b, B_px2_g, B_px2_r])
    return B

def maximization_step(pixel1, pixel2, mean1, mean2, mean3, B):
    Mean_Px1 = []
    Mean_Px2 = []
    Std_Px1 = []
    Std_Px2 = []

    # Calculate the Means and Standard Deviations according to the formula mentioned in the report.
    mean_px1_b = np.sum(np.array(B[0][0]) * np.array(pixel1)) / np.sum(np.array(B[0][0]))
    mean_px1_g = np.sum(np.array(B[0][1]) * np.array(pixel1)) / np.sum(np.array(B[0][1]))
    mean_px1_r = np.sum(np.array(B[0][2]) * np.array(pixel1)) / np.sum(np.array(B[0][2]))
    std_px1_b = (np.sum(np.array(B[0][0]) * ((np.array(pixel1)) - mean1) ** (2)) / np.sum(np.array(B[0][0]))) ** (1 / 2)
    std_px1_g = (np.sum(np.array(B[0][1]) * ((np.array(pixel1)) - mean2) ** (2)) / np.sum(np.array(B[0][1]))) ** (1 / 2)
    std_px1_r = (np.sum(np.array(B[0][2]) * ((np.array(pixel1)) - mean3) ** (2)) / np.sum(np.array(B[0][2]))) ** (1 / 2)

    mean_px2_b = np.sum(np.array(B[0][3]) * np.array(pixel2)) / np.sum(np.array(B[0][3]))
    mean_px2_g = np.sum(np.array(B[0][4]) * np.array(pixel2)) / np.sum(np.array(B[0][4]))
    mean_px2_r = np.sum(np.array(B[0][5]) * np.array(pixel2)) / np.sum(np.array(B[0][5]))
    std_px2_b = (np.sum(np.array(B[0][3]) * ((np.array(pixel2)) - mean1) ** (2)) / np.sum(np.array(B[0][3]))) ** (1 / 2)
    std_px2_g = (np.sum(np.array(B[0][4]) * ((np.array(pixel2)) - mean2) ** (2)) / np.sum(np.array(B[0][4]))) ** (1 / 2)
    std_px2_r = (np.sum(np.array(B[0][5]) * ((np.array(pixel2)) - mean3) ** (2)) / np.sum(np.array(B[0][5]))) ** (1 / 2)

    Mean_Px1.append([mean_px1_b, mean_px1_g, mean_px1_r])
    Mean_Px2.append([mean_px2_b, mean_px2_g, mean_px2_r])
    Std_Px1.append([std_px1_b, std_px1_g, std_px1_r])
    Std_Px2.append([std_px2_b, std_px2_g, std_px2_r])

    return Mean_Px1, Mean_Px2, Std_Px1, Std_Px2



def EM_GMM_algorithm(path,color):
    pixel1 = []
    pixel2 = []
    images = []
    row_list = [["SN", "Mean1", "Mean2", "Mean3", "Std1", "Std2", "Std3"]]

    # Load all the images from respective dataset
    for image in os.listdir(path):
        images.append(image)
    # Iterate over every image in the dataset.
    for image in images:

        image = cv2.imread("%s%s" % (path, image))

        # This block crops out the extra black region and obtains only the buoy region.
        x, y, c = np.where(image != 0)
        TL_x, TL_y = np.min(x), np.min(y)
        BR_x, BR_y = np.max(x), np.max(y)
        image = image[TL_x:BR_x, TL_y:BR_y] # Final cropped image.

        # Extract Green channel of the cropped image and store all its pixels in a list.
        image1 = (image[:, :, 1])
        r, c = image1.shape
        for j in range(0, r):
            for m in range(0, c):
                im1 = image1[j][m]
                if im1 == 0:
                    pass
                else:
                    pixel1.append(im1)

        # Extract Red channel of the cropped image and store all its pixels in a list.
        image2 = (image[:, :, 2])
        r, c = image2.shape
        for j in range(0, r):
            for m in range(0, c):
                im2 = image2[j][m]
                if im2 == 0:
                    pass
                else:
                    pixel2.append(im2)


    n= 0
    num_of_iterations = 50

    # Generate Random numbers in range 0-255
    range_list = list(range(0,256))
    random.shuffle(range_list)
    choice = [range_list.pop(),range_list.pop(),range_list.pop()]
    choice.sort()

    # Initialize the means and standard deviations
    mean1 = choice[1]
    mean2 = choice[0]
    mean3 = choice[2]
    std1 = 10
    std2 = 10
    std3 = 10

    # Train the data for the number of iterations specified
    while (n != num_of_iterations):
        print("n=",n)

        # Perform the Expectation Step.
        B = expectation_step(pixel1,pixel2,mean1,mean2,mean3,std1,std2,std3)

        # Perform the Maximization Step
        Mean_px1, Mean_px2, Std_px1, Std_px2 = maximization_step(pixel1,pixel2,mean1,mean2,mean3,B)

        n = n + 1

        # Update the Mean and Standard Deviation values according to which buoy is being detected.
        if color == "Green":
            mean1 = Mean_px1[0][0]
            mean2 = Mean_px1[0][1]
            mean3 = Mean_px1[0][2]
            std1 = Std_px1[0][0]
            std2 = Std_px1[0][1]
            std3 = Std_px1[0][2]

        if color == "Orange":
            mean1 = Mean_px2[0][0]
            mean2 = Mean_px2[0][1]
            mean3 = Mean_px2[0][2]
            std1 = Std_px2[0][0]
            std2 = Std_px2[0][1]
            std3 = Std_px2[0][2]

        if color == "Yellow":
            mean1 = (Mean_px1[0][0] + Mean_px2[0][0]) / 2
            mean2 = (Mean_px1[0][1] + Mean_px2[0][1]) / 2
            mean3 = (Mean_px1[0][2] + Mean_px2[0][2]) / 2
            std1 = (Std_px1[0][0] + Std_px2[0][0]) / 2
            std2 = (Std_px1[0][1] + Std_px2[0][1]) / 2
            std3 = (Std_px1[0][2] + Std_px2[0][2]) / 2

        print(mean1, mean2, mean3)
        print(std1, std2, std3)

        row_list.append([n, mean1, mean2, mean3, std1, std2, std3])

    # Save the Trained data to a CSV file.
    try:
        with open('train_data_'+color+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    except:
        print("Please close the " + 'train_data_'+color+'.csv'+ 'file before running the code.')


    return  mean1, mean2, mean3, std1, std2, std3











