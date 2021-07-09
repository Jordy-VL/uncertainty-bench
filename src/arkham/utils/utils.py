import os
import pandas as pd
import numpy as np
import regex as re
import time
from contextlib import contextmanager
import pickle
from arkham.default_config import MODELROOT, SAVEROOT, DATAROOT


def generate_timestamp():
    from datetime import datetime

    return str(datetime.now().strftime("%y%m%d-%H%M%S"))


def generate_out_folder(data_folder):
    out_folder = os.path.join(SAVEROOT, "{}".format(data_folder.split("/")[-1]))
    if os.path.exists(out_folder):
        out_folder = os.path.join(
            os.path.dirname(out_folder), generate_timestamp() + "_" + os.path.basename(out_folder)
        )
    return out_folder


def generate_experiment_folder(current, identifier, return_path=False):
    if identifier == "":
        return os.path.join(MODELROOT, os.path.basename(current))
    identifier = identifier.replace("/", "-")
    out_folder = os.path.join(MODELROOT, "{}".format(identifier))
    if os.path.exists(out_folder):
        out_folder = os.path.join(
            os.path.dirname(out_folder), generate_timestamp() + "_" + os.path.basename(out_folder)
        )
    if not return_path:
        os.rename(current, out_folder)
    return out_folder


@contextmanager
def stopwatch(start, stop):
    print("starting: ", start, "at ", time.ctime())
    t0 = time.time()
    yield
    print(stop, "done in ", round(time.time() - t0, 6), ' s')


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(name, "done in ", round(time.time() - t0, 6), ' s')


def num_plot(iterable, key, plot=False):
    df = pd.DataFrame(iterable, columns=[key])
    attr_data = df[key]

    # Compute basic attribute summaries
    min = attr_data.min()
    mean = attr_data.mean()
    median = attr_data.median()
    max = attr_data.max()
    std_dev = attr_data.std()
    perc = np.percentile(attr_data, [25, 75])

    stats = (
        key
        + '\n'
        + 'Min: '
        + str(min)
        + '   '
        + 'Avg: '
        + str(round(mean, 2))
        + '   '
        + 'Std.dev: '
        + str(round(std_dev, 2))
        + '   '
        + 'Median: '
        + str(median)
        + '   '
        + 'Max: '
        + str(max)
        + '   '
        + 'Q1: '
        + str(perc[0])
        + '   '
        + 'Q3: '
        + str(perc[1])
        + '   '
    )
    print(stats)
    print("Suggested cut-off for ", key, ": ", int(mean + std_dev))

    if plot:
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator
        import seaborn as sns

        sns.set(color_codes=True)
        f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.9, 0.1)})
        # Plot the data using countplot
        distplot = sns.distplot(attr_data, ax=ax_hist)

        distplot.set_title(stats)
        distplot.xaxis.set_major_locator(MaxNLocator(integer=True))
        boxplot = sns.boxplot(attr_data, ax=ax_box)
        boxplot.set_xlabel(' ')
        plt.show()


def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if not re.search("=+>", line))
    return '\n'.join(lines)


def pickle_loader(filepath):
    with open(filepath, 'rb') as input_file:
        objfile = pickle.load(input_file)
    return objfile


def pickle_dumper(dumping, path="./pickle_files", filename="data.pickle"):
    import pickle

    with open(os.path.join(path, filename), 'wb') as output_file:
        file = pickle.dump(dumping, output_file, protocol=4)
        print('\npickle saved:\n{}\n'.format(output_file))
    return output_file


def tokenize(lang):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)

    tensor = tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer


def sklearn_split(X, y, test_size, stratified=True):
    seed = 42
    from sklearn.model_selection import train_test_split

    indices = list(range(0, len(y)))
    if stratified:
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y, indices, test_size=test_size, shuffle=True, random_state=seed, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y, indices, test_size=test_size, shuffle=True, random_state=seed
        )
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def stratify_trainvaltest(df, x_col, y_col, validation_size=0.15, test_size=0.2):
    """
    Split a dataframe in train, validation, test dataframes
    :param df:
    :param x_col:
    :param y_col:
    :param validation_size:
    :param test_size:
    :param seed:
    :return:
    """

    def applies_for_stratification(y, n=20):
        # more than half of the category counts below
        out = True
        # n = y.shape[0]
        counts = np.unique(y, return_counts=True)
        counter = dict([(arr, float(counts[1][index])) for index, arr in enumerate(counts[0])])
        satisfactory = len([cat for cat, count in counter.items() if count > n])
        if satisfactory < n / 2:
            out = False
        return out

    stratified = True  # applies_for_stratification(df[y_col].values, n=20)
    X_train_temp, X_test, y_train_temp, y_test, train_temp_indices, test_indices = sklearn_split(
        df[x_col], df[y_col], test_size, stratified=stratified
    )
    train_temp = df.loc[train_temp_indices]
    mapping = {i: train_temp_indices[i] for i in range(0, len(train_temp_indices))}
    if validation_size > 0:
        validation_size *= df.shape[0] / train_temp.shape[0]
        X_train, X_validation, y_train, y_validation, train_indices, validation_indices = sklearn_split(
            train_temp[x_col], train_temp[y_col], validation_size, stratified=stratified
        )
        train_indices = [mapping[x] for x in train_indices]
        validation_indices = [mapping[x] for x in validation_indices]
    else:
        train_indices = train_temp_indices
        X_train, y_train = X_train_temp, y_train_temp
        X_validation, y_validation = [], []
        validation_indices = []
    return X_train, X_validation, X_test, y_train, y_validation, y_test, train_indices, validation_indices, test_indices
