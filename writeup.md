# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

For this final project of term 1 I decided to go with a more structured codebase. The core modules are implemented in the .py files and can be easily re-used in other projects. With each .py core module comes an .ini ConfigParser file which holds the parameters for algorithm tweaking. 

In the Jupyter notebooks I tested the code and played around with the parameters till I was satisfied. The final parameters are than saved in the .ini file for use in the downstream pipeline. The writeup and discussion is included in these Jupyter notebooks (see links below).

The pipeline consists of the following modules:
- config.py [Jupyter](./config.ipynb): Abstract configurable class to implement the interface to the .ini. Other classes inherent from this base class.
- images.py [Jupyter](./images.ipynb): Interface to load images from the test and train set
- featureextractor.py [Jupyter](./featureextractor.ipynb): Extraction of feature vector from images for use in classifier
- classifer.py [Jupyter](./classifier.ipynb): Image classifier to distinguish between vehicle and non-vehicle. The trained classifiers are saved as pickle file in subfolder ./classifier. Downstream pipeline can simple load a pickled classifier.
- objectfinder.py [Jupyter](./objectfinder.ipynb): Searches for objects in an image by sliding windows and matching the trained classifier
- objecttracker.py [Jupyter](./objecttracker.ipynb): Evaluates the matched windows from ObjectFinder and identifies objects.

The final video and discussion is included in the last Juypter notebook.