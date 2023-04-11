Support document for Python_Tracking_Processor

This processing code uses a number of image processing modules:
cv2 - open computer vision (openCV) was used to read and interpret the images 
pims - python image sequence interpreted the edited images as a stack
trackpy - was used to locate the particles and map each trajectory in the stack
scipy - was used to fit curves and filter images

The code uses a number of functions to break the processing into steps. These functions can be called directly in a python environment or indirectly using the jupyter notebooks: z_Wrapper.ipynb/ z_Variable_Finder.ipynb.
The order of operations is as follows:
1.	The data storage and experiment variables are created by i_Setup.py
2.	The video is sliced and edited by ii_Slicer.py
3.	The particles are located, linked and filtered by iii_Tracker.py
4.	The tracks are analysed by iv_Analysis.py
5.	The tracks are mapped by v_Visuals.py
6.	The results of all tracking experiments are compiled by vi_Compiler.py
To find the best variables for each method it is recommended that the video be run through z_Variable_Finder.ipynb. This code is interactive as it breaks down the slicing, tracking and visualising process.
-	This notebook outputs the selected variables such that they can be imported for bulk processing.
After the best variables have been defined import the pickled variables into z_Wrapper.ipynb.

i_Setup.py - folder_generator, Setup
Setup (FileName, Location, Experiment)
This code creates the folders for storing the processing data and extracts the experiment variables from the filenames. 
folder_generator(floc, fname) - given a folder location (floc) and folder name (fname) this method checks if that folder exists before either making the folder or confirming that the folder exists
The code uses the underscores in the filename to find the particle name (pName), wavelength (wlength) and power.

ii_Slicer.py - Slicer
Slicer (FileName, Location, Experiment, alpha, beta, low_thresh, framelimit, filelimit, flipper, save)
This code uses a FileName (.avi) and Location to decompose a video into frames that have been enhanced for brightness and contrast.
This method loops through a number of frames (framelimit) and a number of files (filelimit).
The code uses the average value of each pixel of the first 10 frames to subtract the background from each frame. This is subtracted as an absolute difference in float.64 to allow for negative values. This turns the background black and light/ dark spots white relative to the background.
The background subtracted image is then enhanced using alpha and beta to adjust f(x) using the following equation:
g(x) = α f(x) + β
The parameters α > 0 and β are often called the gain and bias parameters; sometimes these parameters are said to control contrast and brightness respectively.

Parameter Values	Result
alpha = 1  beta =  0	No change
0 < alpha < 1	Lower contrast
alpha > 1	Higher contrast
-127 < beta < +127	+ Lighter      - Darker

The flipper variable flips the images with a wavelength on 532nm across the horizontal axis. When the wavelength is 532nm: Flipper = true.
The save variable is used to keep a version of the original frame in the frames folder.


iii_Tracker.py - (Locate, Link, Filter)
The locate function uses the size/ brightness of features to locate particle positions. The linking function uses a predictor and adaptive search to link particle locations between frames and find trajectories. The filter function filters the trajectories using track length, brightness, size, eccentricity and total displacement variables.
Locate (FileName, Location, Experiment, pixCon, pSize, mMass, thresh, UF, LF = 0)
In the locate function, pixCon is the pixel conversion - um per pixel for each frame. Locate uses pSize and mMass to locate particles - pSize is the particle size in pixels (must be odd and should be larger than the particle) and mMass is the minimum brightness threshold (integrated brightness). Thresh forces pixels below a cut-off to zero.  LF : UF define the frame range of interest.
Link (FileName, Location, Experiment, pDist, lDist, mem)
In the link function, lDist is the search radius of the linking function. The performance of the linking function is improved by a predictor and adaptive search. The predictor typically uses a particles previous velocity to predict its next velocity. Adaptive search is used to avoid conflict by reducing the search range when too many links appear likely. The memory variable (mem) is the maximum number of frames during which a feature can vanish, then reappear nearby, and be considered the same particle. 0 by default.
Filter (FileName, Location, Experiment, filt, lmass, umass, lsize, usize, shape)
The filter function uses a number of variables to validate a track. The filt variable is the lower limit for number of points per track. lmass/umass = brightness bounds for filtering, where mass means total integrated brightness of the blob and lsize/ usize = pixel size bounds for filtering, size means the radius of gyration of its Gaussian-like profile. The shape = ecc is its eccentricity (1 is circular, (0:1)

iv_Analysis.py - Stats and Analysis
Stats(binN, dataN, Ident, F)
The stats function extracts the filename, name, max, min, mean, std for a given histogram and returns it as a dataframe.

Analysis(FileName, Location, Experiment, Input, center, bWidth, nBins, smoo, ptile, cutoff, roundV, fits)
The analysis function searches for the tracjectory data based on input type (Python, ImageJ, Matlab).
The center and bWidth variables are used to define the lasers position and beam width in um.
The variable nBins defines the number of bins for the probability distribution function while
smoo defines the size of the smoothing region.
Ptile/ cutoff are variables for selecting tracks above a threshold. Ptile defines those tracks as an upper percentage of the total tracks while cutoff uses a target velocity to count the percentage of tracks above that cutoff.
RoundV is the the rounding variable (typically 2 places)
Fits is a boolean variable for fitting the results to Gaussian functions. 

v_Visuals.py - Visuals(FileName, Location, Experiment, LF, UF, edited)
LF, UF = Lower and Upper Frames for Visualizing
Edited is a Boolean variable that defines the image the tracks will be displayed on.
Edited = False (Original Frames)
