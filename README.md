# Digital Image Processing Final Project - Business Card Recognition

- André Moreira Souza - N°USP: 9778985
- Josué Grâce Kabongo Kalala - N°USP: 9770382

## Abstract

This project aims to build a digital business card wallet, as a mobile application. This application basically can take a picture of a business card, automatically extracting information about the card, and saves it in the user's digital wallet. During this process, there are digital image processing techniques, such as (1) image denoising / deblurring, (2) image segmentation, and natural language processing techniques for categorizing the extracted text (name, company, job position, email, location, ...). The application should be able to process and extract information from images with different perspectives and possible presence of undesired objects, such as in the examples below.

## Data Sources

For this project, we will build a collection of images for testing purposes. Initially, the collected images are photos, taken from cellphones, of business cards of companies in São Paulo. Those will be used to test each step of the project.

## Expected development stages

[x] - Corner detection - Detect/approximate corners of the business card of the input image. The user should be able to make adjustments when the detection is not accurate.

[x] - Perspective transform - Some images may be taken from different perspectives. This stage's objetive is to normalize the perspective, using on the corner points of the corner detection step, to enhance further operations.

[x] - Text and character recognition - Recognize the characters present in the text of the business card, and build strings with the sequences of recognized characters.

[ ] - Text categorization for the portuguese language - For each sentence, use NLP methods to categorize it into the following categories: entity, phone number, email, location

### Extra development stages

[ ] - Logo detection

[ ] - Web scraping for more info about the extracted entities

[ ] - Text categorization for the english language

## Examples of input images and outputs

### Example 1

![example1](https://d3ui957tjb5bqd.cloudfront.net/images/screenshots/products/0/5/5062/corporate-engineering-company-business-card-template-preview-3-o.jpg?1356390258)

#### Expected extracted data for example 1

| String | Category |
| :----: | :------: |
| HOLBORN & MOORGATE ENGINEERING | Entity |
| +1 801 566-1800 | Phone |
| +1 801 566-1801 | Phone |
| hme.com | URL |
| 2011 S 1100 E | Unknown (coordinates) |
| Salt Lake City, UT 84106 | Location |

### Example 2

![example2](https://www.onextrapixel.com/wp-content/uploads/2017/03/seed-envelope-business-card-1.png)

#### Expected extracted data for example 2

| String | Category |
| :----: | :------: |
| lush | Entity |
| LAWN + PROPERTY ENHANCEMENT | Entity |
| JON | Entity |
| 248-343-5976 | Phone |
| KAYLEN | Entity |
| 734-552-8728 | Phone |
| 6811| Number |
| CLINTOVILLE ROAD | Location |
| CLARKSTON, MICHIGAN 48348 | Location |
| WWW.LUSHMICHIGAN.COM | URL |

### Example 3

![example3](https://nicolenandrasy.files.wordpress.com/2011/10/back2.jpg)

#### Expected extracted data for example 3

| String | Category |
| :----: | :------: |
| Daniel Whitton | Entity |
| Cell: | Entity |
| 817-228-6401 | Phone |
| Email: | Entity |
| dfwcarpentry@verizon.com | email |
| Website: | Entity |
| www.DFWCarpentry.com| URL |
| Address: | Entity |
| 18011 Bruno Road | Location |
| Justin, TX 76247 | Location |

## Final notes

As the examples of inputs and outputs have shown, the final application must be able to identify the text and categorize it based on the sentences found. The user must be able to edit the recognized strings and categories, when the results are not accurate.

## Partial Report

Until the date of this commit (29/05/2019), we have built the image collection for test purposes, and implemented methods for corner detection, utilizing the Harris Corner Detector and Shi-Tomasi Corner Detector.

For the corner detection, we created functions for conversion from RGB to grayscale, computing image derivatives using the Sobel coefficient, and the corner detection function. We've utilized OpenCV's functions for denoising, and Scikit-image's functions for filtering and thresholding.

## Final Report

 A demonstration of the project, its detailed explanation and results discussion can be seen in this [python notebook](https://github.com/andremsouza/dip-project-business-card-recognition/blob/dev/final_report.ipynb)


## Program Usage

The entire project can be executed using the shell program "shell.py".

### Requirements

The following programs and Python packages need to be installed at the running system, for proper execution of the shell.py script:

- Python packages:
    - [numpy](https://pypi.org/project/numpy/)
    - [imageio](https://pypi.org/project/imageio/)
    - [matplotlib](https://pypi.org/project/matplotlib/)
    - [scipy](https://pypi.org/project/scipy/)
    - [scikit-image](https://pypi.org/project/scikit-image/)
    - [opencv-python](https://pypi.org/project/opencv-python/)
    - [Pillow](https://pypi.org/project/Pillow/)
    - [pytesseract](https://pypi.org/project/pytesseract/)
- External programs:
    - [Tesseract](https://github.com/tesseract-ocr/tesseract/wiki)


The shell supports the following commands:
  - `exit`: finish the program and go out from the shell
  - `samples`: show to the user 9 business cards selected randomly from the sample data set.
  - `selectcard <number>`: select a businness card giving a number between 1 and 9.
  - `run`: perform the process to analise and extract the business card informations.
  - `help`: show the usage help message.

Just run `python3 shell.py`.
Enter these commands:
- `samples`. This will show 9 images randomly selected from the sample dataset.
- `selectcard 4`. This will select the fourth card.
- `run`. This will run all the project steps.
- `exit`. This will exit the shell program.

Notice that you have to close the window which shows the image in order to continue using the shell in the terminal.
 
