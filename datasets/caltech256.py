# This file has the code to process caltech256 dataset

# path example: caltech256/219.theodolite/219_0010.jpg
import os
import datautils

def get_images(QUERY):
    '''
    Returns a list of image paths and the category (sorted by category) that it belongs to
    and returns category stats (category #: count)
    '''
    labels           = []
    dataset          = []
    image_paths      = list(datautils.list_images("caltech256"))
    stats = {}
    for (i, image_path) in enumerate(image_paths):
        label = int(image_path.split(os.path.sep)[-1].split("_")[0])
        if(label not in stats):
            stats[label] = 1
        else:
            stats[label] += 1
        dataset.append([image_path, label])
    dataset = sorted(dataset, key = lambda x: x[1])

    training_images = []
    query_images    = []
    dataset_ptr     = 0
    for cat in range(1, 257 + 1): # 257 labels
        print("Splitting category#", cat)
        num_queries = int(stats[cat] * QUERY)

        start = dataset_ptr
        mid   = dataset_ptr + num_queries
        end   = dataset_ptr + stats[cat]

        query_images.append(dataset[start: mid])
        training_images.append(dataset[mid: end])

        dataset_ptr += stats[cat]


    for cat in range(1, 257 + 1):
        print("Checking for category#", cat)
        print(len(query_images[cat - 1]), len(training_images[cat - 1]), stats[cat])
        assert(len(query_images[cat - 1]) + len(training_images[cat - 1]) == stats[cat])
    print("Splitting done")

    return training_images, query_images

training, query = get_images(0.10)
