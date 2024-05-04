import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pickle import dump
from keras.applications.xception import Xception

from utils import DATASET_IMAGES

def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            features[img] = feature
        return features

#2048 feature vector
features = extract_features(DATASET_IMAGES)
dump(features, open("Output/features.p","wb"))