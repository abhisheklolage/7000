# This file has the code to process caltech256 dataset

# path example: caltech256/219.theodolite/219_0010.jpg
import os
import datautils

def get_images():
    '''
    Returns a list of image paths and the category that it belongs to
    and returns category stats
    '''
    labels           = []
    dataset          = []
    image_paths      = list(datautils.list_images("caltech256"))
    category_stats   = {}
    for (i, image_path) in enumerate(image_paths):
        label = int(image_path.split(os.path.sep)[-1].split("_")[0])
        if(label not in category_stats):
            category_stats[label] = 1
        else:
            category_stats[label] += 1
        dataset.append([image_path, label])
    return dataset, category_stats
