#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, 2)
#     #define x and y dimentional containers for left lane and right lane respectively
    x_l = []
    y_l = []
    x_r = []
    y_r = []
    
    # we need two slope watchdogs to keep filter the poins of left lane and right lane
    # as the image is read vertically inversed, the slopes are negative 
    # for left lane and verse versa for the right one
    slope_v = 1# initial left slope, right slope and slope value with fake values
    #iterate to get all points categorized to left and right
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2==x1:# in case for horizontal line
                continue
            if y2==y1:
                continue
                
            slope_v = (y2-y1)/(x2-x1)
            if slope_v < 0:
                x_l.extend([x1,x2]) #add to left lane x container
                y_l.extend([y1,y2])
            else:
                x_r.extend([x1,x2])
                y_r.extend([y1,y2])
                
    #get 1-d lane coefficients
    coe_l = np.polyfit(x_l,y_l,1)
    coe_r = np.polyfit(x_r,y_r,1)
        
    #get straight line slope
    p1_l = np.poly1d(coe_l)
    p1_r = np.poly1d(coe_r)
        
    # get end points for left lane and right lane
    m1,b1 = coe_l # slope and interception for left lane
    m2,b2 = coe_r # slope and interception for right lane
        
    imgshape = img.shape
    y_upper = imgshape[0]*0.6
    y_lower = imgshape[0]
        
    xl_1 = (y_lower-b1)/m1
    xl_2 = (y_upper-b1)/m1
    xr_1 = (y_lower-b2)/m2
    xr_2 = (y_upper-b2)/m2
        
    #draw line
    cv2.line(img, (np.float32(xl_1), np.float32(y_lower)), (np.float32(xl_2), np.float32(y_upper)), color, thickness)
    cv2.line(img, (np.float32(xr_1), np.float32(y_lower)), (np.float32(xr_2), np.float32(y_upper)), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#load images from folder
images = [] # image containaer
test_image_dir = 'test_images/'
for file_name in os.listdir(test_image_dir):
    img = mpimg.imread(os.path.join(test_image_dir,file_name))
    if img is not None:
        images.append([file_name,img])

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
# Define our parameters for Canny
low_threshold = 50
high_threshold = 150
# Define the Hough transform parameters
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 20 #minimum number of pixels making up a line
max_line_gap = 300    # maximum gap in pixels between connectable line segments
# line_image = np.copy(image)*0 # creating a blank to draw lines on

# This time we are defining a four sided polygon to mask
imshape = images[0][1].shape
vertices = np.array([[(110,imshape[0]-1),
                      (imshape[1]*0.5-30, imshape[0]*0.6), 
                      (imshape[1]*0.5+40, imshape[0]*0.6),
                      (imshape[1]-40,imshape[0]-1)]],
                    dtype=np.int32)

#Find the right vertices
#######################################################
# draw lines over an array
def draw_li(img, lines, color=[0, 255, 0], thickness=2):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
        
vert = vertices[0]
ver_m = np.array([[vert[0][0],vert[0][1],vert[1][0],vert[1][1]],
                    [vert[1][0],vert[1][1],vert[2][0],vert[2][1]],
                   [vert[2][0],vert[2][1],vert[3][0],vert[3][1]]])
# draw_li(im_show,ver_m)
#########################################################

# show image
def show_image(im):
    if im is not None:
        if isinstance(im, list):
            for img in im:
                plt.imshow(img)
                plt.show()
        else:
            plt.imshow(im)
            plt.show()
    else:
        print('No image!')       
# imag = images[0]

def draw_lane(img):
    # Read in and grayscale the image
    gray = grayscale(img)
    # apply Gaussian smoothing
    blur_gray = gaussian_blur(gray,kernel_size)
    # apply Canny for color filter
    edges = canny(blur_gray, low_threshold, high_threshold)
    # apply region selection
    masked_edges = region_of_interest(edges, vertices)
    # make a copy of image 
    im_show = np.copy(img)
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # Create a "color" binary image to combine with line image
#     color_edges = np.dstack((edges, edges, edges)) 
        # Draw the lines on the edge image
#     lines_edges = weighted_img(color_edges,line_image) 
    lines_edges = weighted_img(im_show,line_image) 
    return lines_edges
    
for pair in images:
    filename, image = pair
    im = np.copy(image)
    img = draw_lane(im)
#     draw_li(lines_edges,ver_m)
#     saved_filename = 'lane_lined_'+filename
#     image.save('test_images_output/'+saved_filename)
    # show_image(img)
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    img = np.copy(image)
    result = draw_lane(img)
    # you should return the final output (image where lines are drawn on lanes)
    return result



white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(os.path.join(os.getcwd(),white_output), audio=False)
