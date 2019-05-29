# Digital Image Processing Final Project - Business Card Recognition

- André Moreira Souza - N°USP: 9778985
- Josué Grâce Kabongo Kalala - N°USP: 9770382

## Abstract

This project aims to build a digital business card wallet, as a mobile application. This application basically can take a picture of a business card, automatically extracting information about the card, and saves it in the user's digital wallet. During this process, there are digital image processing techniques, such as (1) image denoising / deblurring, (2) image segmentation, and natural language processing techniques for categorizing the extracted text (name, company, job position, email, location, ...). The application should be able to process and extract information from images with different perspectives and possible presence of undesired objects, such as in the examples below.

## Data Sources

For this project, we will build a collection of images for testing purposes. Initially, the collected images are photos, taken from cellphones, of business cards of companies in São Paulo. Those will be used to test each step of the project.

## Expected development stages

[ ] - Corner detection

[ ] - Text line detection

[ ] - Character recognition

[ ] - Text categorization for the english language

### Extra development stages

[ ] - Logo detection

[ ] - Web scraping for more info about the extracted entities

[ ] - Text categorization for the portuguese language

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