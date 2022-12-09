<h1 align="center">Post-Natural Prototype - "Plant Futures" 🌱 </h1>

## Author of this code and respective readme file: 
Praveen Chandrasekaran (pc2846@g.rit.edu) 

## Objective:
The main goal of this code is to analyse the "Luciferase-Pruebas" plant images. This code crunches a dataset containing 378 usable images to cluster them into different groups. The raw images are of size 2832X 4240 and are colored. 

## Resources used in Architecture:
Since the goal of this project is to establish communication between plants, AI and human beings using images, the main software packages revolve around computer vision and some code to handle the sound messages. Google’s drive and Colab were used for most of the experimentation workloads, modeling and testing. The list of software frameworks and resources used for storage, compute, data preprocessing and modeling are shown in table below.

[<img src='https://github.com/Praveen2795/Plant_futures/blob/main/Resources_table.png' alt='Resources' height='400'>

## Architecture Overview:
The architecture that we have created is used to determine the cluster ID of the new/unseen image of the plant. To build communication system that is more robust, we have modeled three independent unsupervised machine learning models on three different types of images: bioluminescent, fluorescent, and thermal. Every machine learning model is different because of its parameters and input images. The machine learning models are trained to cluster the given input images into three groups, so that this information can be used to determine the cluster ID for a given image. ML models are planned to be configured for a continuous learning procedure that uses the new image that doesn’t belong to any cluster-ID. The inference from the models will be used to extract the sound vector for producing a sound which is aimed to make a plant "talk" or "sing". The system architecture can be seen in figure below.

[<img src='https://github.com/Praveen2795/Plant_futures/blob/main/system_flowchart.png' alt='Resources' height='400'>

## Interaction of Resources in Architecture:
A high resolution DSLR camera is used to capture around 500 HD images for bioluminescent, fluorescent and thermal of size 2832X4240 respectively. The camera was configured using Python’s PTP library to automatically capture time lapse photographs and were stored in a local machine to be imported into Google Drive. The usable raw images were selected after a manual screening and cleaning procedures. The filtered images were stored in google drive and later were imported to the Cobal environment as a zip file. Since the number of images was not so big, we chose Google Drive for our operations. Inorder to preprocess the raw images, we used Tensorflow’s img_to_array, load_img and reshape functions. The raw images were resized to 255X255, so that the images are compatible with VGG16 architecture, this architecture was imported from Keras API and the reason for choosing VGG16 is because, this architecture has only 16 trainable layers and the number of parameters are lesser when compared to other popular models like Resnet or Google’s InceptionV3. The light weight VGG16 is more suitable for our use case as the final model is planned to be deployed in an embedded system. 4096 features are extracted from the images using transfer learning, the curse of dimensionality is handled by Principal Component Analysis which is imported from Sklearn API. The number of final features were decided by performing a statistical analysis to know the percent of variance preserved by the features. Lower number of dimensions will help the clustering algorithm train better, for this step PCA was imported from Sklearn’s Decomposition class and was used to reduce the dimensions. The next step in the process is to use the unsupervised clustering algorithm. K-Means algorithm was imported from the Sklearn package to cluster the features into 3 for each image type. The main reason for choosing an unsupervised learning approach is that we wanted the system to spot the patterns and changes in the given features which is far more accurate and efficient than manual classification process. The number of clusters of features would be decided using the popular elbow method on K-Means.

## About the Data:
A high resolution DSLR camera was used to capture around 500 HD bioluminescent images. The camera was configured using Python’s PTP library (code not included) to automatically capture time lapse photographs and were stored in a local machine to be imported into Google Drive. The usable raw images were selected after a manual screening and cleaning procedures. The filtered images were stored in google drive and later were imported to the Cobal environment as a zip file. Since the number of images was not so big, we chose Google Drive for our operations. 

## Luciferase Dataset Link:
<a href="https://drive.google.com/file/d/1AaWdXBl30SXSVFS93YPT0BUPvs7eXwwn/view?usp=sharing">Google drive link</a>

## Prerequisites
To start using this project, you’ll need to either use the [Colb Link]([https://git-scm.com/download/](https://colab.research.google.com/drive/1qIdsA2dlP0R0bp2cDoQY-CYo0-5tA3kN?usp=sharing)) to access the google colab notebook to run the code or have access to the following program. <br/>
- [Python 3.9.9](https://www.python.org/downloads/) <br/>
- [Juptyer Notebook](https://jupyter.org/install) <br/>

## Libraries and APIs:
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://numpy.org/">Numpy</a>
- <a href="https://matplotlib.org/stable/tutorials/introductory/pyplot.html">Matplotlib Pyplot</a>
- <a href="https://scikit-learn.org/stable/">Scikit-learn</a>
- <a href="https://www.tensorflow.org">TensorFlow</a>

## Important Methods Used:
- <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16">VGG-16 for Transfer Learnign</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">PCA for dimensionality reduction</a>
- <a href="https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array">img_to_array</a>
- <a href="https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img">load_img</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html">K-Means</a>

## Getting started

### Installation

#### Python and Jypter notebook
If you have not installed Python 3.9 already, the easiest method to install both the programs is by installing [anaconda](https://www.anaconda.com/) The following link provides a graphical installer link for both Windows and Mac OS [Link](https://www.anaconda.com/products/individual)

If you have already installed Python 3.9 and are an advanced user, you can install Jypter Notebook on terminal by following the steps below.

	pip3 install jupyter	

### Setting Up Local Environment
Navigate to file on google colab and click the download button in found in the dropdown, then click download as .ipynb file to the respective directory where you want to this notebook to be saved in your machine. <br/>

### Running Environments

#### Jypter notebook

If you have installed the Jypter notebook via anaconda, you can run the notebook directly by double clicking on the Jypter notebook icon on the start menu on Windows or Mac, going to the app drawer and selecting the Jypter notebook.

If you have installed Jypter notebook via Python, you can go to Terminal and type the following code:
 
   ```
   jupyter notebook
   ```

Both the processes will open up the Jypter notebook environment in a default web browser. 

In the application window, there will be a file explorer. Navigate to the respective folder where you have cloned the project and select the **" ClusterPruebas.ipynb "** to load up the notebook. 

#### Google colab

If you want to run this project on Google Colab, Navigate to the following [link](https://colab.research.google.com/drive/1qIdsA2dlP0R0bp2cDoQY-CYo0-5tA3kN?authuser=1#scrollTo=nh8qtDZijLqq). Please use this [data link](https://drive.google.com/file/d/1AaWdXBl30SXSVFS93YPT0BUPvs7eXwwn/view?usp=share_link) to downlaod the zipped dataset and upload it into your google drive account. When you are running the code in your google colab, your drive has to be mounted using the below link to access the dataset that you have uploaded. 

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```
After mounting your google drive, you should be able to locate the path of the dataset in your drive and use the below code to unzip the file.
   ```
   import zipfile
   local_zip = '/path.zip'
   zip_ref = zipfile.ZipFile(local_zip, 'r')
   zip_ref.extractall()
   zip_ref.close()
  ```

## Execution and Explanation of Code:
- The recommended way is to access the shared google colab notebook with the above mentioned link and then run the code one by one. 
- Please not note the google colab notebook is documented, nevertheless, exteded explanations are given in this readme file. The code starts with importing essential libraries and APIs that includes but not limited to os, numpy, zipfile, pickle, pandas, matplotlib and tensorflow.
- The next step downloads the zipped folder of dataset into the google colab from the drive. Later unzipping the folder to access the folders inside the main directory.
- The following step is important to understand and check if all the images are accessible, we find 9 folders and some images stored in all of them. We count the number of images present in each of the folders so that we have a total number of files to work with.
- The next cell displayes two images from each folder to see if there are any patterns in brightness of the plants (bioluminescence). We can observe thatthe brightness increases till a point and then starts to fall when the images are displayed in the captured order.
- Next step in the process is to understand our dataset better. We have to preprocess all the images into a size that is acceptable by the model that we will be using for feature extraction. The VGG 16 model is imported from the TensorFlow model zoo.
- Follwoing cell helps in preprocssing the images which are in raw form (size: 2832X4240). These images are not accepted by the VGG16 model. Therefore we resize the input images into 224X224 NumPy arrays. VGG16 mode takes in batches of images rather than a single one. Therefore we reshape the image as (1, 224, 224, 3). Below function takes one of our image and object of our instantiated model to preprocess the image and to return the features. The output layer of the VGG model is removed so that the new final layer is a fully-connected layer with 4,096 output nodes. This vector of 4,096 numbers is the feature vector that we will use to cluster the images.
- After the sanoty checks and preprocessign of data, the next cell helps in addressing the "Curse of Dimensionality" with the help of PCA. This method is used to reduce the number of features from 4096 to a smaller number. We statistically analyse the features to pick a small number for the dimensionality reduction. Typically, we want the explained variance to be between 95–99% from the below graph we statistically find out that 50 or 25 components will be the best for reduction. The chosen number of reduced feature is 50.
- The next step in the process is to use K-Means alogrithm for unsupervised clustering as recognizing subtle changes in bioluminescence is a perfect usecase. We import K-Means from Scikit-learn API.
- After the finalizing the model and its performance, we downlaod the model as a pickle file using Python's util   library. This model.pkl file will be used in the future to pair up the OSC client to send messages. 


## Data Usage Policy:
Data is owned by Rochester Institute of Technology and by professor Carlos Castellanos (carlos.castellanos@rit.edu). Please read the following conditions:

 - The user may not use the dataset wihtout contacting the project members mentioned above.

 - The user may not redistribute the data without separate permission.

 - The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from the faculty member (mentioned above) of Rochester Institute of Technology.

<p align="right">(<a href="#top">back to top</a>)</p>
