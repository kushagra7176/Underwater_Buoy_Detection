import cv2
import numpy as np
from matplotlib import pyplot as plt
import math




def gaussian(x, mu, sig):
    return ((1 / (sig * math.sqrt(2 * math.pi))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))





def compute_Gaussian_Equations(mean_b, mean_g, mean_r, std_b, std_g, std_r, color):
    x = list(range(0, 256))
    gaussian_list = []

    # Convert trained model parameters to numpy arrays to make it compatible with X values.
    mean_b = np.array([mean_b])
    mean_g = np.array([mean_g])
    mean_r = np.array([mean_r])
    std_b = np.array([std_b])
    std_g = np.array([std_g])
    std_r = np.array([std_r])

    # Compute the Gaussian representation for the B, G and R channels of the respective buoy color.
    gaussian_b = gaussian(x, mean_b, std_b)
    gaussian_list.append(gaussian_b)

    gaussian_g = gaussian(x, mean_g, std_g)
    gaussian_list.append(gaussian_g)

    gaussian_r = gaussian(x, mean_r, std_r)
    gaussian_list.append(gaussian_r)

    # Plot the respective Gaussian representations for visualization
    if color == 'Yellow':
        colour = 'y'
    if color == 'Orange':
        colour = 'r'
    if color == 'Green':
        colour = 'g'
    plt.figure(1)
    plt.plot(gaussian_b, color=colour)
    plt.plot(gaussian_g, color=colour)
    plt.plot(gaussian_r, color=colour)

    plt.show()

    plt.figure(2)
    plt.plot(gaussian_b + gaussian_g + gaussian_r, color=colour)
    plt.show()

    # Print the peak values of the respective gaussian representations.
    print("Max value of gaussian representation of blue channel:", max(gaussian_b))
    print("Max value of gaussian representation of green channel:", max(gaussian_g))
    print("Max value of gaussian representation of red channel:", max(gaussian_r))

    return gaussian_list


def detect_All_Three_Buoys(frame, gaussian_yellow, gaussian_orange, gaussian_green, out):
    frame_b, frame_g, frame_r = cv2.split(frame)

    # output_Frame = np.zeros(frame.shape, dtype=np.uint8)
    detect_frame_green = np.zeros(frame_g.shape, dtype=np.uint8)
    detect_frame_yellow = np.zeros(frame_g.shape, dtype=np.uint8)
    detect_frame_orange = np.zeros(frame_g.shape, dtype=np.uint8)

    gaussian_orange_b = gaussian_orange[0]
    gaussian_orange_g = gaussian_orange[1]
    gaussian_orange_r = gaussian_orange[2]
    gaussian_yellow_b = gaussian_yellow[0]
    gaussian_yellow_g = gaussian_yellow[1]
    gaussian_yellow_r = gaussian_yellow[2]
    gaussian_green_b = gaussian_green[0]
    gaussian_green_g = gaussian_green[1]
    gaussian_green_r = gaussian_green[2]


    height, width, ch = frame.shape
    for i in range(height):
        for j in range(width):
            pixel_r = frame_r[i][j]
            pixel_g = frame_g[i][j]
            pixel_b = frame_b[i][j]


            # Condition to detect Orange Buoy:
            if gaussian_orange_r[pixel_r] > 0.252 and pixel_b < 150:
                detect_frame_orange[i][j] = 255
            else:
                detect_frame_orange[i][j] = 0

            # Condition to detect Yellow Buoy:
            if (gaussian_yellow_r[pixel_r] + gaussian_yellow_r[pixel_g])/2 > 0.035 and (gaussian_yellow_g[pixel_r] + gaussian_yellow_g[pixel_g])/2 < 0.014 and pixel_b < 130:
                detect_frame_yellow[i][j] = 255
            else:
                detect_frame_yellow[i][j] = 0

            # Condition to detect Green Buoy:
            if (gaussian_green_r[pixel_g] + gaussian_green_g[pixel_g] + gaussian_green_b[pixel_g] )/3 > 0.018 and (gaussian_green_g[pixel_g] + gaussian_green_b[pixel_g] )/2 < 0.01 and (gaussian_green_g[pixel_g] + gaussian_green_b[pixel_g] )/2 < 0.022/3 and 180 < pixel_r < 200:
                detect_frame_green[i][j] = 255

            else:
                detect_frame_green[i][j] = 0


    ret, threshold_orange = cv2.threshold(detect_frame_orange, 240, 255, cv2.THRESH_BINARY)
    ret, threshold_yellow = cv2.threshold(detect_frame_yellow, 240, 255, cv2.THRESH_BINARY)
    ret, threshold_green = cv2.threshold(detect_frame_green, 240, 255, cv2.THRESH_BINARY)


    # Morphological Operations to detect Orange Buoy:
    kernel1 = np.ones((4,4), np.uint8)
    kernel2 = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0]], dtype=np.uint8)
    kernel3 = np.ones((6, 7), np.uint8)
    dilation_orange_1 = cv2.dilate(threshold_orange, kernel1, iterations=5)
    erosion_orange_1 = cv2.erode(dilation_orange_1, kernel2, iterations=8)
    dilation_orange_2 = cv2.dilate(erosion_orange_1, kernel3, iterations=2)

    # Morphological Operations to detect Yellow Buoy:
    kernel4 = np.ones((4, 4), np.uint8)
    dilation_yellow_1 = cv2.dilate(threshold_yellow, kernel4, iterations=4)

    # Morphological Operations to detect Green Buoy:
    kernel5 = np.ones((4, 4), np.uint8)
    kernel6 = np.array([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]], dtype=np.uint8)
    erosion_green_1 = cv2.erode(threshold_green, kernel6, iterations=1)
    dilation_green_1 = cv2.dilate(erosion_green_1, kernel5, iterations=6)


    # Extract all the contours from the respective buoy images.
    contours_orange, _ = cv2.findContours(dilation_orange_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(dilation_yellow_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(dilation_green_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw circle around the Orange buoy detected.
    for contour in contours_orange:
        num_contours = len(contours_orange)

        if num_contours > 1:
            max_contour_area = max([cv2.contourArea(contour) for contour in contours_orange])

            if cv2.contourArea(contour) < max_contour_area:
                continue

        if 20 < cv2.contourArea(contour):
            (x_orange, y_orange), radius_orange = cv2.minEnclosingCircle(contour)
            center_orange = (int(x_orange) - 2, int(y_orange) - 6)                      # y-axis offset = 6
            radius_orange = int(radius_orange) - 5                                       # radiuse offset to adjust for dilation = 7

            if radius_orange < 15:
                radius_orange = 15

            if radius_orange > 10:
                cv2.circle(frame, center_orange, radius_orange, (0, 79, 255), 2)

    # Draw circle around the Yellow buoy detected.
    for contour in contours_yellow:
        num_contours = len(contours_yellow)
        if num_contours > 1:
            max_contour_area = max([cv2.contourArea(contour) for contour in contours_yellow])

            if cv2.contourArea(contour) < max_contour_area:
                continue
        if 20 < cv2.contourArea(contour):
            (x_yellow, y_yellow), radius_yellow = cv2.minEnclosingCircle(contour)
            center_yellow = (int(x_yellow) - 2, int(y_yellow) - 6)
            radius_yellow = int(radius_yellow) - 5

            if radius_yellow < 17:
                radius_yellow = 15

            if 10 < radius_yellow:
                cv2.circle(frame, center_yellow, radius_yellow, (0, 204, 255), 2)


    # Draw circle around the Green buoy detected.
    for contour in contours_green:
        num_contours = len(contours_green)

        if num_contours < 2:

            if 324 < cv2.contourArea(contour) < 560:
                (x_green, y_green), radius_green = cv2.minEnclosingCircle(contour)
                center_green = (int(x_green), int(y_green) + 1)
                radius_green = 13

                if 8 < radius_green < 17:
                    cv2.circle(frame, center_green, radius_green, (0, 255, 0), 2 )

    cv2.imshow("OUTPUT ",frame)
    out.write(frame)
    if cv2.waitKey(1)==27:
        out.release()
        cv2.destroyAllWindows()



def buoy_Detection(Gaussian_list):

    gaussian_yellow = Gaussian_list[0]
    gaussian_orange = Gaussian_list[1]
    gaussian_green = Gaussian_list[2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Final_Output_Video.avi', fourcc, 5.0, (640, 480))

    input_video_PATH = 'detectbuoy.avi'
    cap = cv2.VideoCapture(input_video_PATH)
    if (cap.isOpened()== False):
        print("Error opening video stream or file. Please place the video file in the Code directory.")

    while cap.isOpened():

        # Read each frame from the video.
        ret,frame = cap.read()

        # Close the stream at the end of video file.
        if not ret:
            break

        detect_All_Three_Buoys(frame, gaussian_yellow, gaussian_orange, gaussian_green, out)

    cap.release()
    cv2.destroyAllWindows()

