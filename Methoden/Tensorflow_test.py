# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 22:03:32 2024

@author: MK
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model = keras.models.load_model("/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Models/c_rt-g_s_o-as_n_100epochs_001.h5")
testpath = "/Users/MK/Documents/GitHub/FOM-Anwendungsprojekt/Data/Output/Airbnb_Prices_V1.0_Test.csv"
testdata = pd.read_csv(testpath, sep=';', decimal='.')
x_t = testdata[['cleanliness_rating', 'guest_satisfaction_overall', 'AttractionScore_Norm']].values
y_t = testdata[['realSum']].values
predictions = model.predict(x_t)
print(predictions)