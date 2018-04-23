# Poster Classifier

Use transfer learning to classify posters into genres with pretrained ResNet.

The [dataset](https://www.kaggle.com/neha1703/movie-genre-from-its-poster/version/3) comes from Kaggle.

### Current Progress:

We have preprocessed the labels into labels.csv.

### Notes:

You can download the poster images [here](https://www.kaggle.com/neha1703/movie-genre-from-its-poster/version/3/discussion/35485). We have checked that those poster images correspond to MovieGenre.csv. But soon enough we will be importing the images into a DataLoader object, and won't need the raw images anymore.