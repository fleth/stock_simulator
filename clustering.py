# -*- coding: utf-8 -*-
import sys
import pandas
import numpy
import glob
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from argparse import ArgumentParser

sys.path.append("lib")
import utils
from loader import Loader

parser = ArgumentParser()
parser.add_argument("start_date", type=str)
parser.add_argument("end_date", type=str)
args = parser.parse_args()

start_date = utils.format(args.start_date)
end_date = utils.format(args.end_date)

# データ読み込み
codes = Loader.high_performance_stocks()

stock_data = []
for code in codes:
    data = Loader.load(code, start_date, end_date)
    if data is not None:
        d = data["close"].values.tolist()
        stock_data.append({"code": code, "data": d})

# データ長を揃えたリスト
datas = list(map(lambda x: x["data"], stock_data))
max_length = max(list(map(lambda x: len(x), datas)))
formatted = []
for data in datas:
    length = max_length - len(data)
    d = data + [data[-1]] * length
    formatted.append(d)

# 一次元配列にする
data = numpy.stack(numpy.array(formatted), axis=0)

stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(data)

seed = 0
numpy.random.seed(seed)
ks = KShape(n_clusters=2, n_init=10, verbose=False, random_state=seed)
y_pred = ks.fit_predict(stack_data)

results = {}
for i, cluster in enumerate(y_pred):
    results[stock_data[i]["code"]] =  cluster

results = sorted(results.items(), key=lambda x: x[1])

for k, v in results:
    print(k, v)
