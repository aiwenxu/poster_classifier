import sys
import pickle
import torch
import matplotlib.pyplot as plt

def quick_print(output):

    print(output)
    sys.stdout.flush()

def clamp_probs(probs):

    output = []

    for value in probs:
        if value >= 0.5:
            output.append(1)
        else:
            output.append(0)

    output = torch.Tensor(output)

    return output

def pickle_stat(o, file_name):

    with open(file_name, "wb") as f:
        pickle.dump(o, f)

def load_stats(file_name):

    with open(file_name, "rb") as f:
        return pickle.load(f)

def plot(vector, file_name):
    x = range(len(vector))
    plt.clf()
    plt.plot(x, vector)
    plt.savefig(file_name)