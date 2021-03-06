##Vehicle Detection
###Description of the algorithm used for vehicle detection.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/bbox1.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code for this step is contained in the the files `feature_extration.py`. HOG is implemented as `hog_features` in `feature_extration.py`. 
I started by reading in all the `vehicle` and `non-vehicle` images (cell 9).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and after multiple trial and error the following values were chosen for the HOG

* color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 9  # HOG orientations
* pix_per_cell = 12 # HOG pixels per cell
* cell_per_block = 2 # HOG cells per block
* hog_channel = "ALL"

###Other features

Apart from HOG, I also used two other features -
* Color Histogram (Cell 12)
I take the color histogram of each color channel and concatenate them together. Then I divide the histogram into 32 bins. 

* Spatial Binning (Cell 6)
I do a spatial binning after resizing the image to 32x32.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The code for this step is contained in the the files `train.py`.
I trained a linear SVC in method `train()` using the spatial binning, color histogram and HOG features. The extracted features were normalized before feeding them to the Linear SVC. Before training, I split the data into the train and test set. The test set is used to compute the accuracy of the Linear SVC. The trained model is saved as a pickle file to be used for prediction in later steps.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The code for this step is contained in the the files `sliding_window.py`.
After comparing the size of vehicles with respect to the camera and prediction accuracy, I decided to use the scales of 1 and 2 (window sizes 64 and 128) for optimum detection. Moreover a `Region of Interest` is used to reduce the area that is searched.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
The code for this step is contained in the the files `detect.py`.
Ultimately I searched on two scales - 1 and 2, using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The code for this step is contained in the the files `convert_video.py`.
Here's a [link to my video result](./output_images/project_video.mp4)
The video is available on youtube [Vehicle detection](https://youtu.be/WCD53eLxeS0)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Challenges: Identifying features and thresholds.

Recommendations: The algorithm is dependent on the features chosen and the thresholds selected. This makes it prone to error if the conditions on the road differ vastly from the data used for calculations. Better approach would be to use Deep Learning for identifying the cars.

