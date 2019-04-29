#
# This code is for converting images to embeddings, using pretrained and trained
# models
#

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model
from datasets import caltech256
from datasets import datautils
import numpy as np

def get_embedding(model, img_path, layer_name):

  return embedding


## get dataset splits

training_paths, query_paths = caltech256.get_images(0.10)
print("DONE")

## training code, get model
## pretrained model
model = ResNet50(weights='imagenet')

##
training_feature_vectors = []
query_feature_vectors = []

PATH = 0
CAT  = 1
intermediate_layer_model = Model(inputs=model.input, \
                                 outputs=model.get_layer('avg_pool').output)
for category in range(1, 258):
    ctr  = 0
    print("Starting category #", category)
    for im in training_paths[category]:
        img = image.load_img(im[0], target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        emb = intermediate_layer_model.predict(img_data)
        np_emb = np.array(emb).flatten()
        training_feature_vectors.append(np_emb)
        ctr += 1
    for im in query_paths[category]:
        img = image.load_img(im[0], target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        emb = intermediate_layer_model.predict(img_data)
        np_emb = np.array(emb).flatten()
        query_feature_vectors.append(np_emb)
        ctr += 1
    print("# images in cat", category, "=", ctr)

np.savez("feature-dump.npz", emb_training = training_feature_vectors, emb_query = query_feature_vectors)
