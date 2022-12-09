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
- NA
- NA
- NA



## Data Usage Policy:
Data is owned by Rochester Institute of Technology and by professor Carlos Castellanos (carlos.castellanos@rit.edu). Please read the following conditions:

 - The user may not use the dataset wihtout contacting the project members mentioned above.

 - The user may not redistribute the data without separate permission.

 - The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from the faculty member (mentioned above) of Rochester Institute of Technology.

<p align="right">(<a href="#top">back to top</a>)</p>
