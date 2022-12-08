<h1 align="center">Post-Natural Prototype - "Plant Futures" 🌱 </h1>

## Author of this code and this readme file: 
Praveen Chandrasekaran (pc2846@g.rit.edu) 

## Objective:
The main goal of this code is to analyse the "Luciferase-Pruebas" plant images. This code crunches a dataset containing 378 usable images to cluster them into different groups. The raw images are of size 2832X 4240 and are colored. 

## Resources used in Architecture:
Since the goal of this project is to establish communication between plants, AI and human beings using images, the main software packages revolve around computer vision and some code to handle the sound messages. Google’s drive and Colab were used for most of the experimentation workloads, modeling and testing. The list of software frameworks and resources used for storage, compute, data preprocessing and modeling are shown in table 1.


## About the Data:
A high resolution DSLR camera was used to capture around 500 HD bioluminescent images. The camera was configured using Python’s PTP library (code not included) to automatically capture time lapse photographs and were stored in a local machine to be imported into Google Drive. The usable raw images were selected after a manual screening and cleaning procedures. The filtered images were stored in google drive and later were imported to the Cobal environment as a zip file. Since the number of images was not so big, we chose Google Drive for our operations. 

### User.dat file description

user_id :: gender :: age :: occupation :: zipcode
				
The data source provided the information that all demographic information is provided voluntarily by the users and is not checked for accuracy.  Only users who have provided some demographic information are included in this data set.

- UserIDs range between 1 and 604 and also serves as the foreign key for User.dat and Ratings.dat file

- Gender is represented by an "M" for male and an "F" for female.

- Age is chosen from the following ranges "Under 18" to "56+" years of age:

- There are 21 different occupations collected by the data source. 

### Movies.dat file description

movie_id :: title :: genres 

- MovieIDs range between 1 and 3952 and also serve as the foreign key for the Movies.dat and Ratings.dat files.

- Titles are identical to titles provided by the IMDB (including year of release)

- Genres are pipe-separated | in the same column:

- Some MovieIDs do not correspond to a movie due to accidental duplicates entries and/or test entries.

- Movies are mostly entered by hand, so errors and inconsistencies may exist.

### Ratings.dat file description

user_id :: movie_id :: rating :: timestamp

- UserIDs range between 1 and 604 and also serve as the foreign key for User.dat and Ratings.dat files.

- MovieIDs range between 1 and 3952 and also serve as the foreign key for the Movies.dat and Ratings.dat files.

- Ratings are made on a 5-star scale (whole-star ratings only).

- Timestamp is represented in seconds since the epoch as returned by time (2).

- Each user has at least 20 ratings.


## Online dataset link

[Link](https://grouplens.org/datasets/movielens/1m/) - Link to the data set.

## Prerequisites

To start using this project with Git, you’ll need to install or have access to the following program. <br/>
- [Git](https://git-scm.com/download/) <br/>
- [Python 3.9.9](https://www.python.org/downloads/) <br/>
- [Juptyer Notebook](https://jupyter.org/install) or [Google Colab](https://colab.research.google.com/) <br/>


## Libraries
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://numpy.org/">Numpy</a>
- <a href="https://seaborn.pydata.org/">Seaborn</a>
- <a href="https://matplotlib.org/stable/tutorials/introductory/pyplot.html">Matplotlib Pyplot</a>
- <a href="http://amueller.github.io/word_cloud/">Wordcloud</a>


## Method used:
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">Tfidf Vectorizer</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html">Linear kernel</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html">Cosine similarity</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">Train test split</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html">Pairwise distances</a>
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html">Mean squared error</a>



## Getting started

### Installation

#### Git
Git can be installed using a CLI or an executable file. The installation instructions can be found at the following link: [Windows](https://git-scm.com/download/win) or [Mac OSX](https://git-scm.com/download/mac)


#### Python and Jypter notebook
If you have not installed Python 3.9 already, the easiest method to install both the programs is by installing [anaconda](https://www.anaconda.com/) The following link provides a graphical installer link for both Windows and Mac OS [Link](https://www.anaconda.com/products/individual)

If you have already installed Python 3.9 and are an advanced user, you can install Jypter Notebook on terminal by following the steps below.

	pip3 install jupyter	

### Setting up your local
In Terminal, navigate to the respective directory where you want to clone this repository and run the following command. <br/>

   ```
   git clone https://github.com/Praveen271195/Movie-Recommedation-System-DRAFT
   ```

If you have forked this repository, use the following code but replace the [username] in the link with your git username.

   ```
   git clone https://github.com/[username]/Movie-Recommedation-System-DRAFT
   ```

### Running the environment

#### Jypter notebook

If you have installed the Jypter notebook via anaconda, you can run the notebook directly by double clicking on the Jypter notebook icon on the start menu on Windows or Mac, going to the app drawer and selecting the Jypter notebook.

If you have installed Jypter notebook via Python, you can go to Terminal and type the following code:
 
   ```
   jupyter notebook
   ```

Both the processes will open up the Jypter notebook environment in a default web browser. 

In the application window, there will be a file explorer. Navigate to the respective folder where you have cloned the project and select the **" Movie Recommendation system - DSCI Final Project.ipynb "** to load up the notebook. 

#### Google colab

If you want to run this project on Google Colab, Navigate to the following [link](https://colab.research.google.com/)

In the application window, click on File->Open Notebook from the menu tab or press Ctrl + O. On the open popup model, click on upload file and go to the respective folder where this project is cloned and select the **" Movie Recommendation system - DSCI Final Project.ipynb "** so that it can be uploaded to Google Colab.

### Running the project
 
1. In Menu -> select Cell -> Run all 

### Interaction

For content-based model search for the following code snippet like mentioned below.

    content_based_test_list = ["Jumanji (1995)", "Pinocchio (1940)", "Walking Dead, The (1995)", "Othello (1995)"]
    
For collaborative-filtering model search for the following code snippet like mentioned below.

    user = [
            {'title':'Breakfast Club, The (1985)', 'rating':4},
            {'title':'Toy Story (1995)', 'rating':2.5},
            {'title':'Jumanji (1995)', 'rating':3},
            {'title':"Pulp Fiction (1994)", 'rating':4.5},
            {'title':'Akira (1988)', 'rating':5}
         ] 

## Data usage license:
Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:

 - The user may not state or imply any endorsement from the
   University of Minnesota or the GroupLens Research Group.

 - The user must acknowledge the use of the data set in
   publications resulting from the use of the data set
   (see below for citation information).

 - The user may not redistribute the data without separate
   permission.

 - The user may not use this information for any commercial or
   revenue-bearing purposes without first obtaining permission
   from a faculty member of the GroupLens Research Project at the
   University of Minnesota.

If you have any further questions or comments, please contact GroupLens
<grouplens-info@cs.umn.edu>.

## Citation:
To acknowledge use of the dataset in publications, please cite the following
paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872


<p align="right">(<a href="#top">back to top</a>)</p>
