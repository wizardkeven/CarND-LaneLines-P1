# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

#### 1. First version for solidWhiteRight and solidYellowLeft
  
The first version for my pipeline consisted of 5 steps:
* Convert the images to grayscale
* Apply Gaussian smoothing
* Apply Canny for color filter
* Apply region selection
* Run Hough on edge detected image
* Create a "color" binary image to combine with line image

As I came into the project right after the previous tutorials, I was thinking that I had been equipped with all the techniques and code to detect lane lines. So I just naively copied all code from Hough Transform Quiz and replaced all necessary part with the given helper functions. Well, it worked so badly on the given images and the output was just a messy. In several seconds, I realized what was going wrong with the previouse well working code but currently poorly working code. Yes, **nearly all the parameters!**

Then, I started adjust the parameters one by one.

First of all, I need to determine which parameter need calibration. 
1.  _**Grayscale**_ has no parameter and I can leave it.
2.  _**Gaussian_blur**_ has one parameter _**kernel_size**_ which I think has been optimized( maybe not, but this will not make determinant effect on the final image I think).
3. _**Canny**_: after examining several grayscaled and gaussian blurred images, I found that the thresholds work quite well for edge detection, so no need to modify.
4. _**Region selection**_: in the example picture, the polygone I applied fitted well for the test picture, but it is not applicable in a relatively real driving environment since it overfitted the given lane lines scale. So I adjusted the vertices of quadrangle to fit the test images.
5. _**Hough transform**_: rho and theta has been determined as unit precision for line detection. In my opinion, It will be not beneficial to increase(more tolerant admission) or decrease(downscaled to less than the unit precision), so I just keep them. * _threshold_（ minimum number of votes，intersections in Hough grid cell）*_min_line_length_（minimum number of pixels making up a line）*_max_line_gap_（maximum gap in pixels between connectable line segments）

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
