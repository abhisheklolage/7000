# Abhishek Lolage 
# ablo9510@colorado.edu

## Datasets
Caltech-256 

## File info
### Colab Notebooks
features.ipynb\
        Notebook to create feature vectors of images from a Deep Learning Model
        in Keras like ResNet50. Also contains code for storing plain embeddings, 
        and ground truth labels of images.

similarity-quadsketch.ipynb \
        Notebook to generate and save answer to 10% queries of each of the
        categories.
similarity-grid.ipynb \
        Notebook to generate and save answer to 10% queries of each of the
        categories.
results\_quadsketch.ipynb contains eval for Plain Embeddings and Quadsketch
results\_grid.ipynb contains evaluation for Grid

### Offline files Sketching
qs.cpp \
The QuadSketch. Originally from Tal Wagner, Ilya Razenshteyn, Piotr Indyk's
paper. Modified to work with custom high dimensional vectors with a bridge for
numpy.

qsrand.cpp \
Randomized variant of the QuadSketch where random bits are used after
pruning when the maximum depth is reached.

grid.cpp \
This is a simple baseline where the space is represented by certain landmarks.
Points in the space are approximated, and represented in compact codes using
these landmarks as reference.  

### Other
    - Intermediate numpy arrays are saved as npz files (Colab <=> Local)

### Dependencies
numpy
cnpy, for c++ python numpy bridge
keras
