
# coding: utf-8

# In[174]:

## Import libraries
import cv2
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.pyplot import figure
#%matplotlib inline


# ### Define Image Processing Functions

# In[171]:


## Applies Preprocessing Operations - 1) Resizes Image 2) Converts to Grayscale 3) Inverts the Image
def scaling(image):

    # Image resised to half coz - Resolution too high and doesnot help much
    resize_status = ""
    resize_percent = 0
    h,w,d = image.shape
    if(h>2000 and w >2000):
        resize_status = "SHRINK"
        resize_percent = 33
    elif(h<300 and w<300):
        resize_status = "ENLARGE"
        resize_percent = 200
    else:
        resize_status = "NOTHING"
        resize_percent = 100


    new_height = int(image.shape[0]*resize_percent/100)
    new_width = int(image.shape[1]*resize_percent/100)

    new_dim = (new_height,new_width)

    # For shrinking Image
    if(resize_status == "SHRINK"):
        image = cv2.resize(image,new_dim,interpolation = cv2.INTER_AREA)
        print("Image Shrinked",new_dim)
    # For enlarging image
    elif(resize_status == "ENLARGE"):
        image = cv2.resize(image,new_dim,interpolation = cv2.INTER_LINEAR)
        print("Image Enlarged",new_dim)
    else:
        print("No Resizing done")

    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale")
    # Inverse the image
    img_inverted = cv2.bitwise_not(img_gray)
    print("Inverting the image..")
    return(img_inverted)


# In[144]:


## Removes Noise - Using Bilateral Filter
def denoising(image):

    # Filtering - Bilateral Filter
    blur = cv2.bilateralFilter(image ,9,75,75)

    print("Denoising the Image..")
    #blur = cv2.fastNlMeansDenoising(blur,None,7,7,21)

    #blur = cv2.GaussianBlur(image,(5,5),0) - blurs the image - so didn't use

    # Morphological Transformations - Dilation
    #kernel = np.ones((5,5),np.uint8)

    # erosion + dilation - doesn't hep either
    #erosion = cv2.erode(blur,kernel,iterations=1)
    #dilation = cv2.erode(erosion,kernel,iterations=1)

    # Only dilation - doesn't help
    #dilation = cv2.dilate(blur,kernel,iterations=2)
    return(blur)
    #return(dilation)


# In[118]:


## Plots two Images
def plot_images(img_baseline,img_improved):
    figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(121),plt.imshow(img_baseline,aspect="auto"),plt.title('Before')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_improved,aspect="auto"),plt.title("After")
    plt.xticks([]), plt.yticks([])
    plt.show()


# In[145]:


### Rotation
def rotation(denoised_image):
    thresh = cv2.threshold(denoised_image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords =  np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    print("Image Rotated by angle of ",angle)
    (h, w) = denoised_image.shape[:2]
    center = (w // 2, h // 2)

    # Get matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(denoised_image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return(rotated)


# In[146]:


# Find the contours and rectangle, and crops the image
def cropping(rotated_image):
    # finding contours in the rotated image

    #contours = cv2.findContours(rotated_image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    edged = cv2.Canny(rotated_image, 10, 50)
    contours = cv2.findContours(edged ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # finding bounding rectangle using contours data points
    rect = cv2.boundingRect(contours[0])
    pt1 = (rect[0],rect[1])
    pt2 = (rect[0]+rect[2],rect[1]+rect[3])
    cv2.rectangle(rotated_image,pt1,pt2,(100,100,100),thickness=2)

    # extracting the rectangle
    text = rotated_image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    print("Cropping the image..")
    return(text)


# ### Apply Image Transformations on an image and generate final image to be ingested in OCR System

# In[105]:


# Read images
#image = cv2.imread('./data/m8.jpg') ## Lorem ipsum
#image = cv2.imread('./data/b2651611_Page_17_60x60-15.png') ## Arabic
#image = cv2.imread('./data/jx8MPc9.png') # French


# In[147]:


#### Perform following operations in order
# Step 1 Preprocessing

def preprocessing(image):
    scaled_image = scaling(image)

    # Step 2 Denoising
    denoised_image = denoising(scaled_image)

    # Step 3 Deskew Images
    rotated_image = rotation(denoised_image)

    # Step 4 Cropping
    text =  cropping(rotated_image)

    text_normal = cv2.bitwise_not(text)

    return(text_normal)


# In[177]:


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,help="path to input image")
args = vars(ap.parse_args())
print(args)


# In[185]:


## Read Input file from input folder
## Write output file to output folder
file = args["image"].split("/")[1]
file_name = file.split(".")[0]
file_type = file.split(".")[1]
# input_image = cv2.imread(os.path.join(os.getcwd()+'/input/'+ 'm8.jpg')) ## Arabic
input_image = cv2.imread(args["image"])
output_image  = preprocessing(input_image)
print("Image Preprocessing Completed !!")
if not os.path.exists(os.getcwd()+'/output'):
    os.makedirs(os.getcwd()+'/output')
output_path = (os.getcwd()+'/output')

file_path = os.path.join(output_path,file_name+"_converted"+"."+file_type)
cv2.imwrite(file_path,output_image)
print("File for Input to OCR ---> ",file_path)
