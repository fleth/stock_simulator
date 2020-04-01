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
parser.add_argument("filename", type=str)
args = parser.parse_args()

try:
    f = open("%s" % args.filename, "r")
    data = json.load(f)
except:
    data = None


if data is None:
    print("%s is invalid json" % args.filename)
    exit()

sum_gain = sum(list(map(lambda x: x["gain"],data.values()))) # 総利益
sum_trade = sum(list(map(lambda x: x["trade"],data.values()))) # 総トレード数
ave_trade = numpy.average(list(map(lambda x: x["trade"],data.values()))) # 平均トレード数
gain_per_trade = sum_gain / sum_trade # 1トレード当たりの利益

average_line = [i*ave_trade * gain_per_trade for i in range(1, len(data)+1)]

gain = 0
diff = []
for ave, d in zip(average_line, sorted(data.items(), key=lambda x:utils.to_datetime(x[0]))):
    gain = gain + d[1]["gain"]
    diff = diff + [abs(ave - gain)]

score = 1 / sum(diff) 

print(score)
