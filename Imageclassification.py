#import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.models import load_model

#import pathlib

from ModelService import ModelService
dataset_url = "https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri/download"
dataset_name = 'brain'
model_name = 'brain.hdf5'


img_url= "file:///Users/mac/Desktop/5c28ee5e8b5e46bb03ef08a8db5ef4_big_gallery.jpg"
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


ModelService.GenerateModel(dataset_url,dataset_name, model_name)
#ModelService.Predict(img_url,class_names,model_name)
#ModelService.Visual(model_name)
