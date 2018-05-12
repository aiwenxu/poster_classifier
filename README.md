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

### Training results:

- 0: lr = 5e-04, lrd = 0.8, best_val_acc = 0.404200.
- 1: lr = 5e-04, lrd = 0.9, best_val_acc = 0.404400.
- 2: lr = 1e-04, lrd = 0.8, best_val_acc = 0.408200.
- 3: lr = 1e-04, lrd = 0.9, best_val_acc = 0.403400.
- 4: lr = 5e-05, lrd = 0.8, best_val_acc = 0.400000.
- 5: lr = 5e-05, lrd = 0.9, best_val_acc = 0.405200.

### Testing results:

We use the best validation accuracy model, i.e., the network parameters saved from the training instance 2 to calculate the final testing accuracy.
The final test accuracy is 0.4033258173618940.