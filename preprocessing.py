import pandas as pd

# This returns ['Short', 'Thriller', 'Family', 'Adventure', 'Documentary', 'Musical', 'Mystery', 'Western', 'Romance',
# 'Animation', 'Game-Show', 'Horror', 'Biography', 'Film-Noir', 'Reality-TV', 'Fantasy', 'Action', 'History', 'News',
# 'Adult', 'Talk-Show', 'Sport', 'Crime', 'Music', 'Sci-Fi', 'War', 'Comedy', 'Drama']
def get_all_genres(data_file):
    genre_set = set([])
    for index, row in data_file.iterrows():
        genre = row["Genre"]
        if isinstance(genre, str):
            genre_set.update(genre.split("|"))
    return list(genre_set)

def get_label(genre, genre_list):
    label = ""
    genre = genre.split("|")
    for possible_genre in genre_list:
        if possible_genre in genre:
            label += "1"
        else:
            label += "0"
    return label

def get_all_labels(data_file, data_dict, genre_list):
    for index, row in data_file.iterrows():
        imdb_id = row["imdbId"]
        genre = row["Genre"]
        if isinstance(genre, str):
            label = get_label(genre, genre_list)
            data_dict["imdb_id"].append(imdb_id)
            data_dict["label"].append(label)

def main():

    data_file = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
    genre_list = get_all_genres(data_file)
    data_dict = {"imdb_id": [], "label": []}
    get_all_labels(data_file, data_dict, genre_list)
    pd.DataFrame(data=data_dict).to_csv("labels.csv", index=False)

if __name__ == '__main__':
    main()