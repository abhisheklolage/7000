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
from numpy import linalg as LA
import sklearn

## gloabls
true_training_categories   = []
true_query_categories      = []
true_training_features     = []
true_query_features        = []
sketched_training_features = []
sketched_query_features    = []

true_answers = []
sketched_answers = []

DIM       = 2048
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
    print(len(true_query_features))
    print(len(true_query_categories))
    print(true_query_categories[0])
    ## for every query compute the NN
    print("Computing for true features...")
    percent = 0
    #for q in range(len(true_query_features)):
    for q in range(300):
        feature = true_query_features[q]
        category = true_query_categories[q]

        min_distance = 1e400
        who = -1     # is the nearest neighbor
        which = -1   # category
        if(q % (len(true_query_features) / 100) == 0):
            print(percent, "% queries done...")
            percent += 10
        for r in range(len(true_training_features)):
            euc_dis = LA.norm(true_training_features[r] - true_query_features[q])
            if(euc_dis < min_distance):
                min_distance = euc_dis
                who = r # index in the database
                which = true_training_categories[r]

        true_answers.append([category, which, r])

    ## sketched feature based nearest neighbor
    print("\nComputing for sketched features...")
    percent = 0
    #for q in range(len(sketched_query_features)):
    for q in range(300):
        feature = sketched_query_features[q]
        category = true_query_categories[q]

        min_distance = 1e400
        who = -1     # is the nearest neighbor
        which = -1   # category
        if(q % (len(sketched_query_features) / 10) == 0):
            print(percent, "% queries done...")
            percent += 10
        for r in range(len(sketched_training_features)):
            euc_dis = LA.norm(sketched_training_features[r] - sketched_query_features[q])
            if(euc_dis < min_distance):
                min_distance = euc_dis
                who = r # index in the database
                which = true_training_categories[r]

        sketched_answers.append([category, which, r])
    print("EVAL")
    print(len(true_answers))
    print(len(sketched_answers))
    print("EVAL")

if __name__ == "__main__":

    ## decompressed
    # dataset =
    # queries =
    ##

    # first entry of both is [], for class indexing 1-257
    true_file = np.load("true-classes.npz")
    train_ex = true_file['tr_paths'] # has first element as []
    query_ex = true_file['qr_paths'] # has first element as []
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
    all_sketched_features_N = [all_sketched_features[i * DIM: (i + 1) * DIM] for i in range(int(len(all_sketched_features)/2048))]
    NUM_QUERY  = len(true_query_features)
    NUM_TRAIN  = len(true_training_features)
    sketched_training_features = all_sketched_features_N[: NUM_TRAIN]
    sketched_query_features = all_sketched_features_N[NUM_TRAIN: len(all_sketched_features_N)]

    print(true_training_features[0])
    print(true_query_features[0])

    #evaluate(1)
    print("Loading sketched vectors...")
