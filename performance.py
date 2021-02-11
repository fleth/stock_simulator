# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import datetime
import numpy
from argparse import ArgumentParser

sys.path.append("lib")
import utils
import slack
import strategy
from loader import Loader


parser = ArgumentParser()
parser = strategy.add_options(parser)
args = parser.parse_args()

try:
    performace_dir="simulate_settings/performances/"
    f = open("%s%sperformance.json" % (performace_dir, strategy.get_prefix(args)), "r")
    data = json.load(f)
except:
    data = None

if data is None:
    print("%s is invalid json" % args.filename)
    exit()

setting_dict, _ = strategy.load_strategy_setting(args)

optimize_end_date = setting_dict["date"]
filtered = list(filter(lambda x: utils.to_datetime(x[0]) < utils.to_datetime(optimize_end_date), data.items()))

sum_gain = sum(list(map(lambda x: x[1]["gain"],filtered))) # 総利益
sum_trade = sum(list(map(lambda x: x[1]["trade"],filtered))) # 総トレード数
ave_trade = numpy.average(list(map(lambda x: x[1]["trade"],filtered))) # 平均トレード数

gain = 0
gains = []
for d in sorted(data.items(), key=lambda x:utils.to_datetime(x[0])):
    gain = gain + d[1]["gain"]
    gains = gains + [gain]

min_gain = min(gains)
gain_per_trade = (sum_gain - (min_gain if min_gain < 0 else -min_gain)) / sum_trade # 1トレード当たりの利益

diff = []
for i, gain in enumerate(gains):
    average = (i+1) * ave_trade * gain_per_trade + min_gain
    diff = diff + [abs(abs(gain) - abs(average))]

score = 1 / sum(diff) 

print(sum(diff))
print(score)
