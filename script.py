import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add ,Bidirectional
import tensorflow.keras.preprocessing
from PIL import Image
import matplotlib.pyplot as plt
from process import gen_chain
import warnings
warnings.filterwarnings("ignore")

##CONSTANTS
max_length=35
model=load_model("captioning.h5")
tokenizer=pickle.load(open("Tokenizer.pkl",'rb'))
features=pickle.load(open("features_flickr.pkl",'rb'))
##VGG MODEL
vgg_model = VGG16() 
vgg_model = Model(inputs=vgg_model.inputs,             
                  outputs=vgg_model.layers[-2].output)



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return 

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text


def generate_caption(image,prompt):
    
    #image = load_img(image_path, target_size=(224, 224))

    # image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    image = preprocess_input(image)
    # extract features
    feature = vgg_model.predict(image, verbose=0)
    # predict from the trained model
    pre=predict_caption(model, feature, tokenizer, max_length)
    print("PRE:",pre)
    pre=' '.join(pre.split()[1:-1])
    gen_caps=gen_chain(pre,prompt)
    print("GEN:",gen_caps)
    return pre, gen_caps


    