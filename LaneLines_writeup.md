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
5. _**Hough transform**_: rho and theta has been determined as unit precision for line detection. In my opinion, It will be not beneficial to increase(more tolerant admission) or decrease(downscaled to less than the unit precision), so I just keep them.
	 * _threshold_（ minimum number of votes，intersections in Hough grid cell）		
	 * _min_line_length_（minimum number of pixels making up a line）
	 * _max_line_gap_（maximum gap in pixels between connectable line segments）.     
I tried to tweak these three values based on the values of the in-class quiz, but always didn't work well while _threshold_ in 5~20, _min_line_length_ in 15~50 and _max_line_length_ in 30~100. I insisted on these ranges as I thought they should work well to determine the small line segments. But as it worked so badly I found myself in the mire. So I decided to get some hints on internet, Luckily, I found [this](https://github.com/naokishibuya/car-finding-lane-lines) . It worked very well when I tried the values 20,20,300. Definitely, I didn't realize a large _max_line_gap_ can help leave the potential dotted lane lines with large interval between two segements. 

Then I need to draw solid lines.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 
applying **numpy.polyfit()** to get 1-d lane coefficients, **numpy.poly1d()** to get straight line slope and finally use the derived to get lanes lines and draw on original RGB image.

Finally, apply code on videos. 

The code worked quite decent on first two videos but completely failed on the challenge video. In face this was under my estimation because the challenge video was taken on a complex light environment and a speedy driving. Lane lines in most frames are just faded by strong sun light or in a shady area under trees. Traditional processing method is inevitably not suitable in this situation. I must find a different way to handle it.

As I referrd to opencv tutorials and Internet, I was enlightened by the idea of color space transformation, especially from RGB to HSV or to HLS. As I tried the two on the captured imges from challenge video, HLS works better on both white lane line and yellow lane line in the most difficult light environment. So I took HLS and retried, Wow, this time, I got a pretty decent result.

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be the parameters of Hough Transform which is probably the cause of some disordered lane lines in solidyYellowLeft.mp4. I have examined the frames which cause the disordered detection, and found they are all in the dotted lane lines with very large interval or the yellow lane lines with cross lines. 

Another shortcoming could be the pre-interpolated initial coordinate values in the function **draw_lines**. I added these fixed initial values for avoiding empty coordinate error during processing of videos which I figured out after encountering this annoying errors thousand times. In certain frames of challenge video or solidYellowLeft video the preprocessed frames could have no valid line segments before _**draw_lines**_ applied. This could help to avoid this bug, but will lead the final "detected lane lines" be the pre-interpolated lines as there is no more other valid lines. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to keep tweaking the parameters of Hough Transform but not excluding others.

Another potential improvement could be to apply color normalization(but the result is not optimistic as I tried. This may need combined with tweaking the parameters at the same time, but I think it worth a try).

_PS: Since tweaking the parameters and solving bugs alongside are quite a time-consuming work, I decide to leave the current code as the final project at time being. I prefer to start the next chapter now and could get back to this project if necessary._
