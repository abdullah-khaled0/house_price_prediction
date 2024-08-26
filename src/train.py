import os
import pickle
import sys

import yaml


def train(seed, n_est, min_split, matrix):
    """
    """


    clf.fit(x, labels)

    return clf


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py features model\n")
        sys.exit(1)


    output = sys.argv[2]
    seed = params["seed"]
    n_est = params["n_est"]
    min_split = params["min_split"]


    clf = train(seed=seed, n_est=n_est, min_split=min_split)

    # Save the model
    with open(output, "wb") as fd:
        pickle.dump(clf, fd)


if __name__ == "__main__":
    main()
