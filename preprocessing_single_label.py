# Single-label preprocessing.

import pandas as pd
import random
from PIL import Image
from helper import quick_print

# This returns ['Adult', 'Music', 'Sport', 'Romance', 'Sci-Fi', 'Thriller', 'Film-Noir', 'History', 'Adventure',
# 'Crime', 'Musical', 'Action', 'Drama', 'Fantasy', 'Family', 'Biography', 'Western', 'Short', 'Mystery', 'War',
# 'Animation', 'Comedy', 'Talk-Show', 'Horror', 'Documentary']. 25 different labels in total.
def get_all_genres(data_file):
    genre_set = set([])
    for index, row in data_file.iterrows():
        genre = row["Genre"]
        if isinstance(genre, str):
            genre_set.add(genre.split("|")[0])
    return list(genre_set)

def get_label(genre, genre_list):
    genre = genre.split("|")[0]
    label = str(genre_list.index(genre))
    return label

def get_all_labels(data_file, data_dict, genre_list):
    for index, row in data_file.iterrows():
        imdb_id = row["imdbId"]
        # Ignore duplicates.
        if imdb_id in data_dict["imdb_id"]:
            continue
        genre = row["Genre"]
        title = row["Title"]
        img_path = "data/posters/{}.jpg".format(imdb_id)
        # Try opening the image files and ignore the ones that can't be opened.
        try:
            img = Image.open(img_path, "r")
            if isinstance(genre, str):
                label = get_label(genre, genre_list)
                data_dict["imdb_id"].append(imdb_id)
                data_dict["title"].append(title)
                data_dict["label"].append(label)
        except OSError:
            quick_print(imdb_id)

def main():

    # Process all the labels.
    data_file = pd.read_csv("data/MovieGenre.csv", encoding="ISO-8859-1")
    genre_list = get_all_genres(data_file)
    data_dict = {"imdb_id": [], "title":[], "label": []}
    get_all_labels(data_file, data_dict, genre_list)
    pd.DataFrame(data=data_dict).to_csv("data/singlelabel/labels.csv", index=False)

    # Split the labels into a training set, a validation set and a testing set.
    # Only the testing set is specified below. The remaining is the training set.
    data_size = 38548
    test_data_size = 3548
    validate_data_size = 5000
    indices = list(range(data_size))
    random.shuffle(indices)
    test_set = indices[:test_data_size]
    validate_set = indices[test_data_size:(test_data_size+validate_data_size)]

    test_data_dict = {"imdb_id": [], "title":[], "label": []}
    validate_data_dict = {"imdb_id": [], "title":[], "label": []}
    train_data_dict = {"imdb_id": [], "title":[], "label": []}

    for i in range(data_size):
        if i in test_set:
            test_data_dict["imdb_id"].append(data_dict["imdb_id"][i])
            test_data_dict["title"].append(data_dict["title"][i])
            test_data_dict["label"].append(data_dict["label"][i])
        elif i in validate_set:
            validate_data_dict["imdb_id"].append(data_dict["imdb_id"][i])
            validate_data_dict["title"].append(data_dict["title"][i])
            validate_data_dict["label"].append(data_dict["label"][i])
        else:
            train_data_dict["imdb_id"].append(data_dict["imdb_id"][i])
            train_data_dict["title"].append(data_dict["title"][i])
            train_data_dict["label"].append(data_dict["label"][i])

    pd.DataFrame(data=test_data_dict).to_csv("data/singlelabel/test_labels.csv", index=False)
    pd.DataFrame(data=validate_data_dict).to_csv("data/singlelabel/validate_labels.csv", index=False)
    pd.DataFrame(data=train_data_dict).to_csv("data/singlelabel/train_labels.csv", index=False)

if __name__ == '__main__':
    main()