

import cv2
import matplotlib.pyplot as plt
import os
import math
import numpy as np


def gaussian(x, mu, sig):
    return ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

PATHS = ["DATASET/Yellow/", "DATASET/Orange/", "DATASET/Green/"]
color = ['Yellow', 'Orange', 'Green']

# Iterate over all the image paths.
for ind,image_PATH in enumerate(PATHS):
    input_flag = False
    list_hist_b = []
    list_hist_g = []
    list_hist_r = []

    list_of_filenames = [filename for filename in os.listdir(image_PATH) ]
    dataset_size = len(list_of_filenames)
    print("dataset size :",dataset_size)

    for i in range(0,dataset_size):
        frame = cv2.imread(image_PATH+list_of_filenames[i])

        # Extract the buoy region while cropping out the black region.
        x, y, c = np.where(frame != 0)
        TL_x, TL_y = np.min(x), np.min(y)
        BR_x, BR_y = np.max(x), np.max(y)
        count = 6
        cropped = frame[TL_x :BR_x , TL_y :BR_y ]

        frame_b,frame_g,frame_r = cv2.split(cropped)

        # Calculate the respective channel histograms and store it ina list.
        hist_b = cv2.calcHist([frame_b],[0],None,[256],[1,256])
        hist_g = cv2.calcHist([frame_g],[0],None,[256],[1,256])
        hist_r = cv2.calcHist([frame_r],[0],None,[256],[1,256])
        list_hist_b.append(hist_b)
        list_hist_g.append(hist_g)
        list_hist_r.append(hist_r)

    #  Calculate the Average histogram from the histogram ist generated above.
    avg_hist_b = [(sum(x)/dataset_size).tolist() for x in zip(*list_hist_b)]
    avg_hist_g = [(sum(x)/dataset_size).tolist() for x in zip(*list_hist_g)]
    avg_hist_r = [(sum(x)/dataset_size).tolist() for x in zip(*list_hist_r)]

    # Plot the average histograms
    fig, axs = plt.subplots(1, 1, figsize=(5, 3) )
    axs.plot(avg_hist_r,color = 'r')
    axs.plot(avg_hist_g,color = 'g')
    axs.plot(avg_hist_b,color = 'b')

    fig.suptitle('Histogram for R, G, B channels of '+ color[ind] + ' buoy')

    fig.savefig("Output_images/Average_histogram_of_"+color[ind]+"_buoy.png")
    fig.show()

    # This block is used to generate a normal distribution over the average histogram calculated above.

    arr_b = frame_b.astype('float')
    arr_g = frame_g.astype('float')
    arr_r = frame_r.astype('float')

    # Convert all the 0's in the image to np.nan
    arr_b[frame_b == 0] = np.nan
    arr_g[frame_g == 0] = np.nan
    arr_r[frame_r == 0] = np.nan


    # Calculate mean and std while ignoring the np.nan values.
    frame_b_mean = np.nanmean(arr_b)
    frame_b_std = np.nanstd(arr_b)

    frame_g_mean = np.nanmean(arr_g)
    frame_g_std = np.nanstd(arr_g)

    frame_r_mean = np.nanmean(arr_r)
    frame_r_std = np.nanstd(arr_r)

    print("mean blue:",frame_b_mean)
    print("std blue:",frame_b_std)

    print("mean green:",frame_g_mean)
    print("std green:",frame_g_std)

    print("mean red:",frame_r_mean)
    print("std red:",frame_r_std)

    x = np.array((range(0, 256))).T
    x = list(range(0, 255))

    # Compute the Gaussian Distribution from the average histogram.
    frame_b_gaussian = gaussian(x, frame_b_mean, frame_b_std)
    frame_g_gaussian = gaussian(x, frame_g_mean, frame_g_std)
    frame_r_gaussian = gaussian(x, frame_r_mean, frame_r_std)


    plt.plot(frame_r_gaussian, color='r')
    plt.plot(frame_g_gaussian, color='g')
    plt.plot(frame_b_gaussian, color='b')


    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()













