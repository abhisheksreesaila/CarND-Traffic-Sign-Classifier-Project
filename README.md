# CarND-Traffic-Sign-Classifier-Project

The goals / steps of this project are the following:
- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

## Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic signs data set:
- Number of training examples = 34799
- Number of testing examples = 12630
- Image data shape = (32, 32)

### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First I checked the random images to explore how the images look. Then curious about the distribution of the data in various classes in train, validation and test set.

![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/explore.png)

### Design and Test a Model Architecture

As a first step, I wanted to perform data augmentation and yield more samples for my training set. Hence used the library “Augmentor” which makes it super easy to create images. The actual training set was used a starting point, random images were chosen and “rotation” was applied. This will help us train the model with more diverse images and won’t let the model overfit. Then I decided to convert the images to grayscale because it’s a common technique used to reduce input dimensions. Also I found that each pixel value “averages out” when you convert to grayscale. This will help in the next step which is normalization where in each pixel values is normalized by the formula
- **(gray_image - gray_image.min()) / (gray_image.max () - gray_image.min ())**

This will ensure the values are scaled down between 0 and 1 which help in model speed, accuracy and convergence.

### Architecture

![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/model.png)

My final model consisted of the following layers: I used a classic LeNet Architecture which accepts 32x32x1 as input and outputs 43 logits.

My final model results were:
- Training loss of almost 0%
- validation set accuracy of  95.7%
- Test set accuracy of 93.2%
A well-known architecture was chosen:
- LeNet Architecture was chosen as a starting point. It worked on MNIST dataset and the traffic sign data had a very similar dataset i.e. multi-classification problem on a set of images with similar size and resolution. 
- How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

### Test a Model on New Images

- **Image 1:** This is clear image with no symbols in the center. It is also unique than other symbols. Should be easily to classify for the model. 

- ![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/image1.jpg)

- **Image 2:** It is also distinct than others symbols. Should be easily to classify for the model. 
- ![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/image2.jpg)
- **Image 3:** This has some text in the middle, it has some noise and the model is not trained on such images. Expected mis-classification
- ![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/image3.jpg)
- **Image 4:** This is big blue background + the sign is of blue color hence it might confuse the model. Chances are 60/40 in favor of misclassification
- ![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/image4.jpg)
- **Image 5:** This looks unique from other symbols model should classify the image correctly.
- ![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/image5.jpg)
### Misclassification Analysis
- ![doc.png](https://github.com/abhisheksreesaila/CarND-Traffic-Sign-Classifier-Project/raw/master/misclassifications.PNG)
