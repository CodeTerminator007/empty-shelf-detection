# Identifying-Empty-Sections-in-Store-Shelf
Image processing algorithm to identify empty sections in store shelves

**Problem Statement**

Given an image, how can we create an algorithm using image processing techniques to
identify the sections where there is no product present.

![Example](https://github.com/sbasu26/Identifying-Empty-Sections-in-Store-Shelf/blob/main/ExamplePic.png)

**Solution**

This approach detected the empty shelves with bounding boxes using Template Matching for Multiple Objects in OpenCV.
I cropped the sample input image to take out an empty shelf, as the Template.

![Template](https://github.com/sbasu26/Identifying-Empty-Sections-in-Store-Shelf/blob/main/Template.png)

This template image was matched with the Input Image

![Input Image](https://github.com/sbasu26/Identifying-Empty-Sections-in-Store-Shelf/blob/main/InputImage.png)

The **Template Matching** technique generated multiple bounding boxes for each empty shelf, hence non-maximum suppression was used on the resulting bounding boxes to select 3 distinct bounding boxes for the 3 empty shelves present in the input image.
Below is the final output image:

![Output Image](https://github.com/sbasu26/Identifying-Empty-Sections-in-Store-Shelf/blob/main/OutputImage.png)

The challenge faced for this approach was identifying the appropriate image processing technique that would give the desired result.
Canny Edge detection, Contours, Houghlines, Histogram Backprojection where some of the tried techniques, but none of these provided a simple approach towards identifying voids between products or product edges.

The code for this approach can be found in the file Template Matching with Non Max Supression.

**Approach 2: Machine Learning / Deep Learning**

Using ML/DL, when we try to detect empty shelves (or voids) between products on a grocery store shelf, we would have to prepare our data so that we have images of:

1. Shelves with voids (or missing products) : Class 1
2. Shelves with No voids (or missing products) : Class 2

After labeling the data, we can perform Object Detection on this data using either of the below techniques:

 HOG with SVM
 R-CNN
 Fast R-CNN
 Faster R-CNN
 YOLO
 SSD

The choice of object detection technique here would warrant the accuracy and time taken for prediction.
Challenges here might be the width of the voids, which might determine the number of predictions per void or gap in the shelf. If 3 products are missing would we want to identify 3 voids for 3 missing products or 1 void for the whole gap between the products.

*******
