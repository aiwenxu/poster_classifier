# Poster Classifier

Use transfer learning to classify posters into genres with pretrained ResNet.

MovieGenre.csv can be downloaded [here](https://www.kaggle.com/neha1703/movie-genre-from-its-poster/version/3).

The poster images are downloaded [here](https://www.kaggle.com/neha1703/movie-genre-from-its-poster/version/3/discussion/35485).

(We have checked that those poster images correspond to MovieGenre.csv. However, some of them cannot be identified by Pillow for some unknown reasons. We ignore those that cannot be identified.)

(Fun story: After deleting the duplicates and those images that cannot be identified, the size of the dataset boils down to 38548.)

### Dependencies:

- PyTorch
- Pandas
- Pillow
- Matplotlib