import tensorflow as tf
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


data_folder = 'data'
recipes_zip = tf.keras.utils.get_file('recipes.zip',                                     
											origin = 'https://www.dropbox.com/s/i1hvs96mnahozq0/Recipes5k.zip?dl=1',
											extract = True)
print(recipes_zip)
data_folder = os.path.dirname(recipes_zip)
os.remove(recipes_zip)
print(data_folder)