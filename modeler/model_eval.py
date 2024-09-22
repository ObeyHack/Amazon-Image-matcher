"""
we will evaluate the model using the test data, the test data will be (image,text) pairs.
we will use this kind of data to evaluate the model because (image,label) does not really make sense in our case.
there can two items in the same category but they can be very different in terms of their text.

1)find images with text
2)get url from our model
3)get the image and text that corresponds to the url
4) check if the text is similar to the text that we have in our data
5) if it is similar, then we can say that our model is working correctly.
"""

import text_evaluation
import imager.vgg.VGG
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(test_data, model):
    # test_data is a list of (image,text) pairs (MAYBE WE (image,url) pairs)
    accuracy = 0
    "WRITE THE CODE HERE"

    return accuracy



