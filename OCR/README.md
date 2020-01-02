# Optical Character Recognition 

## About
The following repository contains the OCR functionality with focus on understanding the **image preprocessing cycle**  
to be applied on image documents to improve the overall accuracy of the OCR Engine (in our case - `Tesseract-OCR`)

First we 1) explain the image preprocessing components and then run some examples to preprocess   
and then 2)run the output file on open source OCR engine Tesseract which recognizes text in the image documents

## Image Preprocessing Components - (`OCR_preprocessing.py`)  

#### 1. Preprocess
   
   * Step 1
   
      * Shrinks the image if the image is too large h>2000,w > 2000, By a factor of two
      * Enlarges the image if the image is too small h<300,w < 300, By a factor of two
   
      * Learnings -  Factoring by 3,4 times were tried but didn't improves the predicted words
   
   * Step 2
   
      * Converts 3 color channel image to Grayscale channel
   

   * Step 3
    
      * Invert the image - Because the input images have white background and black text we inverse their pixel values.
      And therefore change the background to black and text to white for further step
   

#### 2. Denoising

   * Purpose - Removes Salt & Pepper Noise 
   
   * Step 1

      * Applies Bilateral Filter
      It applies two components - Gaussian Filter on the image and filter that takes into account the difference in intensity 
      between neighborhood pixels
   
      Learnings - Gaussian Filter was applied but it resulted in loss of some text along with the noise removal 
   

#### 3. Rotation/Deskewing
   
   * Purpose - To straighten an image 

   * Steps 
   
      * For the inverted image, we get all coordinates which are part of the foreground
      * Then these coordinates are passed to minAreaRect which computes the angle which is between -90 to 0, 
        with -90 means the image is rotated by 90 degrees counterclockwise
      * We apply an affine transformation (cv2.warpAffine) to correct for the skew using the angle and centre coordinates  

#### 4. Cropping

   * Purpose -To crop the text part from the image and exclude the non textual part
    
   * Steps
     
      * We first binarize the image using OTSU's adaptive threshold method
      * We then detect the edges of the text through Canny Edge detection method
      * Then contours are extracted from these edges
      * A rectangle around the 4 extreme coordinates of these contousr are formed,and the image is cropped around that


## How to Run On Images

**1. Preprocessing **  

* Step 1
      * Open Terminal  

* Step 2 
      * Place all the images for which text to be generated in input folder 

* Step 3  
      * To run for a single image, we pass a parameter with the image name      
       --image image_name.extension to the ocr_preprocessor.py    

      * For ex, we run the following command for our 3 images     
      `python ocr_preprocessor.py --image input/m8.jpg`  
      `python ocr_preprocessor.py --image input/b2651611_Page_17_60x60-15.png`  
      `python ocr_preprocessor.py --image input/jx8MPc9.png`  


It will generate output preprocessed files in the output directory with the naming convention  
image_name_converted.extension


**2.OCR using Tesseract Engine**

Run the OCR engine on preprocessed image to recognize text using the tesseract OCR engine

* Step 1  

       * Download Tesseract 4.0.0  
       `sudo yum install epel-release`  
       `yum install tesseract`  

* Step 2  

      * We run the following command to generate text characters from the image using the   
         `tesseract output/image_name_converted.extension output/image_name_converted --oem 3 -l fas --dpi 300`  

      * We can Specify the language of the document with -l   
           For ex `-l eng` will recognize English document  
           Full list of language codes can be accessed by  
           `tesseract -list-langs`

      * For ex, we run the following command for our 3 images  
         1.`tesseract output/m8_converted.jpg output/m8_converted --oem 3 -l eng --dpi 300`  
         2.`tesseract output/b2651611_Page_17_60x60-15_converted.png output/b2651611_Page_17_60x60-15 --oem 3 -l ara --dpi 300`  
         3.`tesseract output/jx8MPc9_converted.png output/jx8MPc9 --oem 3 -l fra --dpi 300`  

      * This saves the following extracted text files in the output folder   
         1.m8_converted.txt    
         2.b2651611_Page_17_60x60-15.txt    
         3.jx8MPc9.txt  

## Learnings 

#### English Text  (`m8.jpg`)
* English text's best accuracy is achieved when it's not resized because the aspect ratio(width to height) ratio is already 
  2.2:1 
* Also if we add fastNlMeansDenoising denoising to english text, as it's already clear,we in turn looses some of the text 
  in recognizing

#### French  (`jx8MPc9.png`)
*  Erosion And Dilation didn't helped - ink got scattered  
*  fastNlMeansDenoising looses some text, instead of helping   
*  Marginal noise i.e the black dark line on the left of the document, is not yet removed, using traditional opencv filters  
   like median,gaussian,bilateral filters

#### Arabic  (`b2651611_Page_17_60x60-15.png`)
*  The image resolution is too large, and when passed through OCR doesn't give any output  
*  We shrink the image by a factor of two  
*  Further rescaling doesn't improves performance  


## Discrepancy and Issues
*  Marginal Noise Still Exist for the French documents, which can be worked upon
*  Arabic Text is not completelt correctly classified even after all the image Preprocessing - Needs Separate OCR steps 


