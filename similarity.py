#
# This code is for doing similarity search on set of images.
# Image features generated using:
# resnet pretrained-embeddings, resnet importance sampling trained
# Sketches (compressed representations) of features generated using
# grid, quadsketch, quadsketch variant
#

## imports
# libraries
import numpy as np
import sklearn

## gloabls
true_training_categories   = []
true_query_categories      = []
true_training_features     = []
true_query_features        = []
sketched_training_features = []
sketched_query_features    = []

## functions

def query(Q, K):
    '''
    Q could be the original or the sketched feature vector
    Returns K nearest images
    '''
    pass

def evaluate(K):
    ## get class of vector

    ## original feature based nearest neighbor

    ## sketched feature based nearest neighbor

    pass

if __name__ == "__main__":

    ## decompressed
    # dataset =
    # queries =
    ##

    # first entry of both is [], for class indexing 1-257
    true_file = np.load("true-classes.npz")
    train_ex = true_file['tr_paths']
    query_ex = true_file['qr_paths']
    for category in range(1, 258):
        for im in train_ex[category]:
            true_training_categories.append(int(im[1]))
        for im in query_ex[category]:
            true_query_categories.append(int(im[1]))

    ## from the colab file, as is, no change required
    true_features_file = np.load("resnet50-pretrained-np-dump.npz")
    true_training_features = true_features_file['emb_training']
    true_query_features = true_features_file['emb_query']

    ## from the cpp codes, requires to segregate training and query
    sketched_features_file = np.load("resnet50-qs-dump.npz")
    print(sketched_features_file.files)
    all_sketched_features = sketched_features_file['qstq']
    all_sketched_features_N = [all_sketched_features[i * 2048: (i + 1) * 2048] for i in range(int(len(all_sketched_features)/2048))]
    NUM_QUERY  = len(true_query_features)
    NUM_TRAIN  = len(true_training_features)
    DIM       = 2048
    sketched_training_features = all_sketched_features_N[: NUM_TRAIN]
    sketched_query_features = all_sketched_features_N[NUM_TRAIN: len(all_sketched_features_N)]

    evaluate(2)
    print("Loading sketched vectors...")
