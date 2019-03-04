import sys
import collections
import subprocess
from argparse import ArgumentParser

sys.path.append("lib")
from loader import Loader
import strategy
from strategy import CombinationSetting

parser = ArgumentParser()
parser.add_argument("count", type=int)
parser.add_argument("-n", "--n_calls", type=int, default=100, action="store", dest="n_calls", help="探索回数")
parser = strategy.add_options(parser)
args = parser.parse_args()

prefix = strategy.get_prefix(args)
filename = strategy.get_filename(args)

combination_setting = CombinationSetting()
combination_strategy = strategy.load_strategy_creator(args, combination_setting)

checker = strategy.CombinationChecker(combination_strategy)

results = {}
for i in range(args.count):
    setting = Loader.simulate_setting("tmp/%s_%s" % (i, filename))
    if setting["score"] == 0:
        print("skip. score zero. %s" % i)
        continue
    for r in checker.get_strategy_sources(setting):
        method = r["key"]
        res = r["combinations"][0] + r["combinations"][1]
        # 選ばれた条件のインデックスを数える
        if method in results:
            results[method] = results[method] + res
        else:
            results[method] = res

for method, combinations in results.items():
    print(method)
    for index, count in collections.Counter(combinations).most_common():
        source = checker.get_strategy_source_by_index(index, checker.combinations_dict[method][1])
        print(" - %s : %s" % (count, source))


