	[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
	#############################################################################################
	#######        ENPM 673 - Perception for Autonomous Robots                           ########
	#######        Authors @Kushagra Agrawal                                             ########
	#######        Project 3 : Underwater Buoy Detection                                 ########
	#############################################################################################


	#########################################################################
	#                    #***************************#                      #
	#                    #*** Directory Structure ***#			#
	#                    #***************************#			#
	#     									#
	#									#
	# 	"kushagra7176.zip" 						#
	# (Extract this compressed file to any desired folder) 			#
	#	Folder Name after extraction: kushagra7176		        #
        # 	|-EM_GMM_Algorithm.py					        #
	#	|-Final_Buoy_Detection.py			       	        #
	#	|-Gaussian_Mixture_Model_for_Buoy_Detection.py		        #
        #	|-GMM_Generate_Avg_Hist.py				        #
	#	|-GMM_Generate_Dataset.py				        #
	#	|-Report.pdf							#
	#	|-README.md							#
	#	|-DATASET							#
	#########################################################################

***************************** Special Instructions ****************************** 
-- Install all package dependencies like OpenCv(cv2) before running the code.
-- Update pip and all the packages to the latest versions.
-- Make sure you have the following files in your directory:
	1. "detectbuoy.avi"
	2. "DATASET" folder
			|- "Orange"
					|- "frame0.png, frame1.png ... "
			|- "Yellow"
					|- "frame0.png, frame1.png ... "
			|- "Green"
					|- "frame0.png, frame1.png ... "
	3.Final_Buoy_Detection.py
	4.EM_GMM_Algorithm.py	

-- To obtain proper output in the 'GMM_Generate_Dataset.py ' file, follow the following steps:

	1. First create a DATASET folder in the directory as the code and inside that create three folders with names 'Green', 'Orange', 'Yellow'
	2. When the .py file is run, the frames from the input video is displayed. Select points on this image that mark the region of interest and then close that 
	   image window to save the cropped image.
	3. Verify that the cropped ROI images are saved in their respective directories in the DATASET folder. 

Instructions to run the file:

################## PyCharm IDE preferred #############

-> Extract the "kushagra7176.zip" file to any desired folder or location.

-> 1. Using Command Prompt:

->-> Navigate to the "Code" folder inside "kushagra7176" folder, right click and select open terminal or command prompt here. (Skip Step 2)
	*** use 'python3' instead of 'python' in the terminal if using Linux based OS ***
	Execute the files by using the following commands in command prompt or terminal:
	--python .\GMM_Generate_Dataset.py 
	--python .\GMM_Generate_Avg_Hist.py
	--python .\Gaussian_Mixture_Model_for_Buoy_Detection.py

-> 2. Using PyCharm IDE

	Open all the files in the IDE and click Run and select the "<filename.py>" to execute.

	
FINAL NOTE: 

->	The 'GMM_Generate_Dataset.py' file is used to generate the DATASET.
->	The 'GMM_Generate_Avg_Hist.py' file is used to generate the average histograms of each buoy.
->	The 'Gaussian_Mixture_Model_for_Buoy_Detection.py' is used to first plot the final trained gaussian representation of all the buoys and then 
	display the final buoy detection output	frame by frame and also generate a final output video file named 'Final_Output_Video.avi'in the same 
	directory as the code.
