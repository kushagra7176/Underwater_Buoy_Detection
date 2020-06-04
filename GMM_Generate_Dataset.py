import cv2
import numpy as np
import matplotlib.pyplot as plt

roi_coordinates = []
list_of_roi_coordinates =[]
frame_count = 0

plt_window_closed_flag = False
input_flag = False

# Function to crop and save ROI image to desired location.
def crop_image(frame_count,image_PATH):

    print("Frame Count:::",frame_count+1)

    # Initialize image mask of original frame dimensions with zeros.
    mask = np.zeros(frame.shape, dtype=np.uint8)

    # Create numpy array of the roi coordinates.
    roi_corners = np.array([roi_coordinates], dtype=np.int32)
    white = (255, 255, 255)

    # Fill the ROI region of mask with white
    cv2.fillPoly(mask, roi_corners, white)

    # Apply mask on original frame.
    masked_image = cv2.bitwise_and(frame, mask)

    # Convert ROI image back to BGR format to be used by Open-CV.
    masked_image = cv2.cvtColor(masked_image,cv2.COLOR_RGB2BGR)

    # Display generated ROI image.
    cv2.imshow("masked", masked_image)

    '''
    Save the image to the destination image path. ( PATH::: For Yellow->>> "DATASET/Yellow/frame" || 
                                                            For Orange->>> "DATASET/Orange/frame" ||
                                                            For Green ->>> "DATASET/Green/frame"  ||)
    '''
    # Change the PATH according to the above instructions.

    cv2.imwrite(image_PATH + str(frame_count) + ".png", masked_image)

# Function to callback when plt window is closed.
def handle_close(evt):
    print('Closed Figure!')
    list_of_roi_coordinates.append(roi_coordinates)
    global plt_window_closed_flag
    plt_window_closed_flag = True

# Function to check if window is closed
def waitforbuttonpress(frame_count,image_PATH):
    while plt.waitforbuttonpress(0.2) is None:
        global roi_coordinates
        if plt_window_closed_flag:
            # cv2.imshow("frame" + str(frame_count), frame)
            try:
                crop_image(frame_count,image_PATH)
            except:
                pass
            roi_coordinates = []
            return True
    return False

# Function to callback when Left mouse button click is detected on the plt window.
def onclick(event):
    if event.xdata != None and event.ydata != None:
        roi_coordinates.append((event.xdata, event.ydata))
        print(event.xdata, event.ydata)



########################################################################################################################
###########################                            MAIN                                  ###########################
########################################################################################################################

# Set PATH to input Dataset video. (Current path set to Code directory.)
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



    # Convert image to RGB format to make it compatible with matplotlib
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Display the image so that ROI can be marked by mouse clicks.
    implot = plt.imshow(frame)

    # Callback function to link function with even manager. (Mouse click event)
    cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
    # Callback function to link function with even manager. (Window close event)
    implot.figure.canvas.mpl_connect('close_event', handle_close)

    # Test point to display fixed number of frames. (Uncomment to use it)
    '''
    test_num_of_frames = 5
    if frame_count > test_num_of_frames-1:
        break
    '''

    while input_flag is not True:
        print("Welcome to Dataset Generator!!")
        dataset_path_color = input("Please enter the buoy color for which the dataset is being generated (Accepted variables:'orange','green' or 'yellow'): ")
        if dataset_path_color == "yellow":
            image_PATH = "DATASET/Yellow/frame"
            input_flag = True
        elif dataset_path_color == "orange":
            image_PATH = "DATASET/Orange/frame"
            input_flag = True
        elif dataset_path_color == "green":
            image_PATH = "DATASET/Green/frame"
            input_flag = True
        else:
            print("Please enter a color from the approved variables mentioned!!")
            input_flag = False


    # Set path to the destination folder according to user input.

    plt.show()
    # Increment frame count when plt window is closed.
    try:
        if waitforbuttonpress(frame_count,image_PATH):
            frame_count = frame_count+1
            # break
    except:
        pass


    if cv2.waitKey(0)==27:
        cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()



