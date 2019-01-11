# -*- coding: utf-8 -*-
import sys
import numpy
import json
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout

sys.path.append("lib")
import utils
import checker
from loader import Loader

# data_checker.pyで残差が定常過程である(ランダムウォークでない)ことを確認していることが前提

args = sys.argv
if len(args) < 3:
    print("using: %s [start_date] [end_date] [code]" % args[0])
    exit()

# start を1/1, endを12/31にしてしまうと前後のデータがないのでロードに失敗する
start_date = utils.format(args[1])
end_date = utils.format(args[2])
code = args[3]

# パラメータ
term = 5

# データ読込
data = Loader.load(code, start_date, end_date)
ols_res, resid = checker.ols(data["close"])

# モデル準備
model = Sequential()
model.add(LSTM(256, batch_input_shape=(None, term, 1)))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# データ準備
resid = preprocessing.scale(resid)
X, Y = [], []
for i in range(len(resid) - term):
    X.append(resid[i:i+term])
    Y.append(1 if resid[i+term] > 0 else 0) #翌日のH&L予測
#    Y.append(resid[i+term]) #翌日値の予測

# データ整形
X = numpy.array(X).reshape(-1, term, 1)
Y = numpy.array(Y)

split_pos = int(len(X) * 0.8)
x_train, x_test = X[0:split_pos], X[split_pos:]
y_train, y_test = Y[0:split_pos], Y[split_pos:]

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=5, nb_epoch=500)

predict = model.predict(numpy.array(resid[-term:]).reshape(-1, term, 1))
loss, accuracy = model.evaluate(x_test, y_test, batch_size=5)
print(json.dumps({
    "predict": str(predict[0][0]),
    "accuracy": accuracy
}))
