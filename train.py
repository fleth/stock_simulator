# -*- coding: utf-8 -*-
import os
import sys
import numpy
import pandas
import random
import json
import shutil
import sklearn.metrics
import autosklearn.classification
import multiprocessing

from sklearn import preprocessing
from sklearn import datasets
from keras.utils import np_utils
from keras.optimizers import Adam
from datetime import datetime, timedelta
from keras import regularizers
from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
import keras.backend as K
import collections

sys.path.append("lib")
import utils
import keras_utils
import strategy
from loader import Loader
from autoencoder import Autoencoder
from trainer import TrainerSetting, Trainer, ModelCreator

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

args = sys.argv
if len(args) < 3:
    print("using: %s [mode] [start_date] [end_date]" % args[0])
    exit()

mode = args[1]
start_date = utils.format(args[2])
end_date = utils.format(args[3])

def load(setting, start_date, end_date):
    # データ準備
    X, Y = [], []

    x_columns = None
    x_columns = ["daily_gradient", "weekly_gradient"]

    stocks = Loader.stocks()
    stock_data = Loader.loads(stocks["code"], start_date, end_date, with_stats=True)
    for code, data in stock_data.items():
        x = data.drop("date", axis=1)
        y = data["entity"] * data["yin"] # 実体と方向
        shape = setting.batch_input_shape if setting.with_batch else setting.input_shape

        if x_columns is not None:
            x = x[x_columns]

        if setting.with_batch:
            x = to_seq(setting, x)
            x = numpy.nan_to_num(x.reshape(shape))
            y = y.iloc[setting.length_of_sequences:]
        else:
            x = numpy.nan_to_num(x.as_matrix().reshape(shape))
        y = numpy.nan_to_num(y.as_matrix())
        X.extend(x)
        Y.extend(y)
    X = numpy.array(X)
    Y = numpy.array(Y)
    print(X.shape, Y.shape)
    return X, Y

def to_seq(setting, data):
    seq = []
    length = setting.length_of_sequences
    for i in range(length):
        r = -(length-(i+1))
        if r == 0:
            d = data.iloc[i:-1]
        else:
            d = data.iloc[i:r-1]
#        print(i, r, len(d))
        seq.append(d.as_matrix())
    seq = numpy.array(seq)
    print(seq.shape, setting.batch_input_shape)
    return seq

# ブースティング用重みファイルのリスト
def weights():
    path = "simulator/weights/boosting"
    if not os.path.exists(path):
        os.mkdir(path)

    files = os.listdir(path)
    return files

# 重みファイルをブースティング用にコピー
def copy_weights(trainer):
    path = "simulator/weights/boosting"
    files = weights()
    shutil.copy(trainer.weights_path(), os.path.join(path, "weights_%s.hdf5" % str(len(files)+1)))

def output_tsv(setting):
    X, Y = load(setting)
    Y = Y.reshape(-1, 1)
    print("x_train", X.shape)
    print("y_train", Y.shape)
    keras_utils.output_csv(X, "x_train.tsv", sep='\t')
    keras_utils.output_csv(Y, "y_train.tsv", sep='\t')

    path = "simulator/results"
    if not os.path.exists(path):
        os.mkdir(path)

    for i, x in enumerate(X):
        print(x)
        keras_utils.output_pgm(x, "%s/x_%s.pgm" % (path, i), 100, 5)

def autoencode(setting):
    setting.hidden_activation = "relu"
    setting.output_activation = "sigmoid"
    setting.loss = "binary_crossentropy"

    autoencoder = Autoencoder(setting)

    X, Y = load(setting)
    keras_utils.output_csv(X, "strategy.csv")
    split_pos = int(len(X) * setting.split)
    x = X[0:split_pos]
    x_test = X[split_pos:]

    autoencoder.train(x, x_test)
    loss, accuracy = autoencoder.evaluate(x)
    print("loss: %s, accuracy: %s" % (loss, accuracy))

def svm(setting):
    X, Y = load(setting)
    x, y, x_test, y_test = keras_utils.validate_split(X, Y, setting.split)

    model = sklearn.svm.SVC()
    model.fit(x, y)
    predicted = model.predict(x_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predicted))

def auto_sk(setting):
    X, Y = load(setting)
    x, y, x_test, y_test = keras_utils.validate_split(X, Y, setting.split)

    automl = AutoSklearnClassifier()
    automl.fit(x, y)
    predicted = automl.predict(x_test)
    print(automl.show_models())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predicted))


def train(setting, model, start_date, end_date, output=True):
    X, Y = load(setting, start_date, end_date)
    x, y, x_test, y_test = keras_utils.validate_split(X, Y, setting.split)

    print(x.shape, y.shape, x_test.shape, y_test.shape)

    trainer = Trainer(setting, model)
    trainer.train(x, y, x_test, y_test, tensorboad=True)

    copy_weights(trainer)

    return eval_accuracy(trainer, x_test, y_test, timestamp=None, output=output)

def boost(setting, model, output=True):
    X, Y = load(setting)
    x, y, x_test, y_test = keras_utils.under_sampling(X, Y, setting.split)

    trainer = Trainer(setting, load_model)
    predict = trainer.predict(x)
    predict = list(map(lambda x: x.argmax(), predict))
    setting.sample_weight = numpy.array(list(map(lambda x: 0.5 if x[0] == x[1] else 2.0, zip(y, predict))))
    trainer = Trainer(setting, model)
    trainer.train(x, y, x_test, y_test, tensorboad=True)

    copy_weights(trainer)

    return eval_accuracy(trainer, x_test, y_test, timestamp=None, output=output)

def bagging(setting, model):
    X, Y = load(setting)

    setting.class_weight = utils.with_class_weight(Y)

    chunked_x = utils.chunked(X, 5000)
    chunked_y = utils.chunked(Y, 5000)

    print("chunk: %s" % len(chunked_x))

    for x, y in zip(chunked_x, chunked_y):
#        x, y, x_test, y_test = keras_utils.under_sampling(x, y, setting.split)
        x, y, x_test, y_test = keras_utils.validate_split(x, y, setting.split)
        trainer = Trainer(setting, model)
        trainer.train(x, y, x_test, y_test)
        eval_accuracy(trainer, x_test, y_test, output=False)
        copy_weights(trainer)


def validate(setting, model):
    X, Y = load(setting)
    x, y, x_test, y_test = keras_utils.under_sampling(X, Y, 0.5)

    trainer = Trainer(setting, load_model)
    return eval_accuracy(trainer, x_test, y_test)


def validate_boost(setting, model):
    files = weights()

    X, Y = load(setting)
    x, y, x_test, y_test = keras_utils.under_sampling(X, Y, 0.5)

    trainer = Trainer(setting, load_model)

    predicts = []
    for f in files:
        trainer.setting.weights_filename = "boosting/%s" % f
        trainer.load_weights()
        predict = trainer.predict(x)
        predict = list(map(lambda x: x.argmax(), predict))
        predicts.append(predict)

    predicts = numpy.array(predicts).T
    predict = []
    for p in predicts:
        count_dict = collections.Counter(numpy.asarray(p))
        max_label = max((v,k) for (k,v) in count_dict.items())[1]
        predict.append(max_label)

    correct(y, predict)

def eval_accuracy(trainer, x, y, timestamp=None, output=True):
    loss, acc = trainer.evaluate(x, y)
    print("Accuracy : %s" % acc)

    predict = trainer.predict(x)
    predict = list(map(lambda x: x.argmax(), predict))

    #correct(y, predict)

    if not output:
        return loss, acc, re, pre

    keras_utils.output_csv(predict, "predict.csv", timestamp)
    keras_utils.output_csv(y, "y_test.csv")

    return loss, acc, re, pre

def correct(y, predict):
    count_dict = collections.Counter(numpy.asarray(y))
    collect_dict = dict()
    for true, pred in zip(y, predict):
      if not true in collect_dict:
        collect_dict[true] = 0
      if true == pred:
        collect_dict[true] += 1

    result = dict()
    for key in count_dict.keys():
        result[key] = (collect_dict[key]/float(count_dict[key])) * 100
    print("#--correct--#")
    print(count_dict, collect_dict)
    print(result) # ラベルごとの正答率
    print(predict[-1])
    print("#--/correct--#")


# ======================================================
if __name__ == "__main__":
  setting = TrainerSetting()
  setting.split = 0.8
  setting.input_neurons = 2
  setting.length_of_sequences = 5
  setting.output_neurons = 1
  setting.hidden_neurons = 128
  setting.filter_size = 128
  setting.batch_size = 500
  setting.nb_epoch = 500
  setting.depth = 0
  setting.dropout = 0.02
  setting.hidden_activation = "relu"
  setting.output_activation = "linear"
  setting.loss = "mean_squared_error"
  setting.optimizers = "adam"
  setting.metrics = ["accuracy"]
  setting.input_shape = (-1, setting.input_neurons)
  setting.batch_input_shape = (-1, setting.length_of_sequences, setting.input_neurons)
  setting.output_shape = (-1)

  setting.categorical = False
  setting.with_batch = True
  model = ModelCreator.lstm

  if mode == "train":
    train(setting, model, start_date, end_date)

  if mode == "boost":
    boost(setting, model)

  if mode == "bagging":
    bagging(setting, model)

  if mode == "autoencode":
    autoencode(setting)

  if mode == "validate":
    validate(setting, model)

  if mode == "validate_boost":
    validate_boost(setting, model)

  if mode == "auto_sklearn":
    auto_sk(setting)

  if mode == "svm":
    svm(setting)

  if mode == "output_tsv":
    output_tsv(setting)
