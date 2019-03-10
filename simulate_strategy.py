# -*- coding: utf-8 -*-
import sys
import pandas
import json
import time
import numpy
import copy
import subprocess
import datetime
from skopt import gp_minimize
from multiprocessing import Pool
from argparse import ArgumentParser

sys.path.append("lib")
import utils
import slack
import strategy
from loader import Loader
from simulator import Simulator, SimulatorSetting, SimulatorData
from strategy_simulator import StrategySimulator

###
# 2018-01-01 ~ 2018-10-10 までを1ヶ月毎評価で最適化する(--outputを指定すると設定出力)
# - python simulator/simulate_strategy.py 2018-10-01 1 -o 10 [--output]
#
# 2018-01-01 ~ 2018-10-10 までを1ヶ月毎評価で出力済み設定でシミュレーションする
# - python simulator/simulate_strategy.py 2018-10-01 1 -c 10

# (最適化期間10ヶ月 検証期間1ヶ月)の2期間分を 1ヶ月毎評価で出力済み設定でシミュレーションする
# - python simulator/simulate_strategy.py 2018-10-01 1 -c 2 -o 10 --ignore_optimize
# ['2017-11-30 00:00:00 - 2018-09-30 00:00:00', '2016-12-30 00:00:00 - 2017-10-30 00:00:00'] 最適化期間
# ['2018-09-30 00:00:00 - 2018-10-30 00:00:00', '2017-10-30 00:00:00 - 2017-11-30 00:00:00'] 検証期間
###
parser = ArgumentParser()
parser.add_argument("date", type=str)
parser.add_argument("validate_term", type=int) # 最適化期間の10~20%
parser.add_argument("--ignore_optimize", action="store_true", default=False, dest="ignore_optimize", help="最適化を実施しない")
parser.add_argument("-o", type=int, action="store", default=0, dest="optimize_count", help="最適化期間の数")
parser.add_argument("-c", type=int, action="store", default=1, dest="count", help="検証期間の数")
parser.add_argument("--code", type=str, action="store", default=None, help="code")
parser.add_argument("--assets", type=int, action="store", default=None, dest="assets", help="assets")
parser.add_argument("-n", "--n_calls", type=int, action="store", default=100, help="simulate n_calls")
parser.add_argument("-j", "--jobs", type=int, action="store", default=8, dest="jobs", help="実行ジョブ数")
parser.add_argument("-v", action="store_true", default=False, dest="verbose", help="debug log")
parser.add_argument("--output", action="store_true", default=False, dest="output", help="設定をファイルに出力")
parser.add_argument("--upload", action="store_true", default=False, dest="upload", help="結果をslackに送信する")
parser.add_argument("--random", type=int, action="store", default=0, dest="random", help="ランダム学習の回数")
parser.add_argument("--auto_stop_loss", action="store_true", default=False, dest="auto_stop_loss", help="自動損切")
parser.add_argument("--stop_loss_rate", type=float, action="store", default=0.02, dest="stop_loss_rate", help="損切レート")
parser = strategy.add_options(parser)
args = parser.parse_args()

# start を1/1, endを12/31にしてしまうと前後のデータがないのでロードに失敗する

def create_setting(args, assets):
    setting = SimulatorSetting()
    setting.min_data_length = args.validate_term * 10
    setting.assets = assets
    setting.commission = 150
    setting.debug = args.verbose
    setting.sizing = False
    setting.short_trade = args.short
    setting.auto_stop_loss = args.auto_stop_loss
    return setting

def create_simulator_data(param):
    code = param["code"]
    start_date = param["start_date"]
    end_date = param["end_date"]
    args = param["args"]

    settings = strategy.LoadSettings()
    settings.with_stats = args.with_stats
    settings.weekly = not args.ignore_weekly

    data = strategy.load_simulator_data(code, start_date, end_date, args, settings)
    return data

def load_index(args, start_date, end_date):
    index = {}

    if args.tick:
        return index # 指標はティックデータない

    for k in ["nikkei"]:
        d = Loader.load_index(k, start_date, end_date, with_filter=True, strict=False)
        index[k] = utils.add_index_stats(d)

    return index

def load(args, codes, terms, daterange):
    min_start_date = min(list(map(lambda x: x["start_date"], terms)))
    prepare_term = utils.relativeterm(args.validate_term, args.tick)
    start_date = utils.to_format(min_start_date - prepare_term)
    end_date = utils.format(args.date)

    print("loading %s %s %s" % (len(codes), start_date, end_date))
    data = {}
    params = list(map(lambda x: {"code": x, "start_date": utils.to_format(daterange[x][0]), "end_date": utils.to_format(daterange[x][-1]), "args": args}, codes))

    p = Pool(8)
    ret = p.map(create_simulator_data, params)
    for r in ret:
        if r is None:
            continue
        data[r.code] = r

    index = load_index(args, start_date, end_date)

    print("loading done")
    return {"data": data, "index": index, "args": args}

def get_score(args, scores, simulator_setting, strategy_setting):
    if args.tick:
        score = get_tick_score(scores, simulator_setting, strategy_setting)
    else:
        score = get_default_score(scores, simulator_setting, strategy_setting)
    return score

def get_score_stats(scores):
    gain = list(map(lambda x: x["gain"], scores))
    win = list(filter(lambda x: x > 0, gain))
    loss = list(filter(lambda x: x < 0, gain))
    drawdown = list(map(lambda x: x["drawdown"], scores))
    max_drawdown = list(map(lambda x: x["max_drawdown"], scores))
    trade = list(map(lambda x: x["trade"], scores))
    win_trade = list(map(lambda x: x["win_trade"], scores))
    profit_factor = sum(win) / abs(sum(loss)) if abs(sum(loss)) > 0 else sum(win)
    gain_per_trade = sum(gain) / sum(trade) if sum(trade) > 0 else 0

    return {
        "gain": gain,
        "win": win,
        "loss": loss,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
        "trade": trade,
        "profit_factor": profit_factor,
        "gain_per_trade": gain_per_trade,
        "win_trade": win_trade
    }

def print_score_stats(name, score, score_stats, assets, strategy_setting):
    stats = [
        "max_drawdown", max(score_stats["max_drawdown"]),
        "min_gain:", min(score_stats["gain"]) / assets,
        "sum_gain:", sum(score_stats["gain"]),
        "pf:", score_stats["profit_factor"],
        "gpt:", score_stats["gain_per_trade"],
        "t:", sum(score_stats["trade"])]

    print(name, stats, score)
    setting = {"name": name, "stats": stats, "score": score, "setting": strategy.strategy_setting_to_dict(strategy_setting)}
    with open("settings/simulate.log", "a") as f:
        f.write(json.dumps(setting))
        f.write("\n")

def get_default_score(scores, simulator_setting, strategy_setting):
    score_stats = get_score_stats(scores)

    ignore = [
#        len(list(filter(lambda x: x > 0.1, score_stats["max_drawdown"]))) > 0, # 最大ドローダウンが10%以上の期間が存在する
#        len(list(filter(lambda x: x < -0.1, score_stats["gain"]))) > 0, # 10%以上の損失が存在する
#        len(list(filter(lambda x: x > 0, gain))) < len(scores) / 2, # 利益が出ている期間が半分以下
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
        sum(score_stats["trade"]) < 30, # 取引数が少ない
        score_stats["profit_factor"] < 1.5, # プロフィットファクター（総純利益 / 総損失）が1.5以下
#        gain_per_trade < 5000, # 1トレードあたりの平均利益が5000円以下
    ]

    score = score_stats["profit_factor"] * sum(score_stats["gain"]) * (1 - max(score_stats["max_drawdown"]))
    if any(ignore):
        score = 0

    print_score_stats("default:", score, score_stats, simulator_setting.assets, strategy_setting)

    return score

def get_tick_score(scores, simulator_setting, strategy_setting):
    score_stats = get_score_stats(scores)

    ignore = [
#        len(list(filter(lambda x: x > 0.1, max_drawdown))), # 最大ドローダウンが10%以上の期間が存在する
#        len(list(filter(lambda x: x > 0, gain))) < len(scores) / 2, # 利益が出ている期間が半分以下
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
#        sum(score_stats["trade"]) < 30, # 取引数が少ない
        score_stats["profit_factor"] < 1.5, # プロフィットファクター（総純利益 / 総損失）が1.5以下
#        gain_per_trade < 5000, # 1トレードあたりの平均利益が5000円以下
    ]

    score = score_stats["profit_factor"] * sum(score_stats["gain"]) * (1 - max(score_stats["max_drawdown"])) * sum(score_stats["win_trade"])
    if any(ignore):
        score = 0

    print_score_stats("tick:", score, score_stats, simulator_setting.assets, strategy_setting)

    return score

# 1つの設定でstart~endまでのterm毎のシミュレーション結果を返す
def simulate_by_multiple_term(strategy_setting, datas, terms, strategy_simulator):
    scores = []
    tick = datas["args"].tick
    for term in terms:
        # term毎にデータを分けてシミュレートした結果の平均をスコアとして返す
        start = utils.to_format_by_term(term["start_date"], tick)
        end = utils.to_format_by_term(term["end_date"], tick)
        stats = strategy_simulator.simulates(strategy_setting, datas, start, end, datas["args"].verbose)
        scores.append(stats)
    return scores

# パラメータ評価用の関数
# 指定戦略を全銘柄に対して最適化
def objective(args, strategy_setting, datas, terms, strategy_simulator):
    print(strategy_setting.__dict__)
    try:
        scores = simulate_by_multiple_term(strategy_setting, datas, terms, strategy_simulator)
        score = get_score(args, scores, strategy_simulator.simulator_setting, strategy_setting)
    except Exception as e:
        print("skip objective. %s" % e)
        import traceback
        traceback.print_exc()
        score = 0
    return -score

def strategy_optimize(args, datas, terms, strategy_simulator):
    print("strategy_optimize: %s" % (utils.timestamp()))
    strategy_setting = strategy.StrategySetting()

    # 現在の期間で最適な戦略を選択
    space = strategy_simulator.strategy_creator.ranges()
    n_random_starts = int(args.n_calls/10) if args.random > 0 else 10
    random_state = int(time.time()) if args.random > 0 else None
    res_gp = gp_minimize(
        lambda x: objective(args, strategy_setting.by_array(x), datas, terms, strategy_simulator),
        space, n_calls=args.n_calls, n_random_starts=n_random_starts, random_state=random_state, n_jobs=args.jobs)
    result = strategy_setting.by_array(res_gp.x)
    score = res_gp.fun
    print("done strategy_optimize: %s random_state:%s" % (utils.timestamp(), random_state))
    return result, score

def create_terms(args):
    terms = []
    validate_terms = []

    valid_end_date = utils.to_datetime(args.date)
    for c in range(args.count):
        end_date = valid_end_date - utils.relativeterm(args.validate_term, args.tick)
        start_date = end_date - utils.relativeterm(args.validate_term*args.optimize_count, args.tick)
        term = {"start_date": start_date, "end_date": end_date}
        validate_term = {"start_date": end_date, "end_date": valid_end_date}
        if args.tick:
            term["start_date"] += datetime.timedelta(hours=9)
            term["end_date"] += datetime.timedelta(hours=15)
            validate_term["start_date"] += datetime.timedelta(days=1, hours=9)
            validate_term["end_date"] += datetime.timedelta(hours=15)
        terms.append(term)
        validate_terms.append(validate_term)
        valid_end_date = start_date
    print(list(map(lambda x: "%s - %s" % (str(x["start_date"]), str(x["end_date"])), terms)))
    print(list(map(lambda x: "%s - %s" % (str(x["start_date"]), str(x["end_date"])), validate_terms)))
    return terms, validate_terms

def create_performance(simulator_setting, performances):
    print(json.dumps(performances))

    # レポート出力
    with open("settings/performance.json", "w") as f:
        f.write(json.dumps(performances))

    # 簡易レポート
    for k, v in sorted(performances.items(), key=lambda x: utils.to_datetime(x[0])):
        print(k, v)

    gain = sum(list(map(lambda x: x["gain"], performances.values())))
    average_trade_size = numpy.average(list(map(lambda x: len(x["codes"]), performances.values())))

    result = {
        "gain": [gain],
        "return": [gain / simulator_setting.assets],
        "max_drawdown": [max(list(map(lambda x: x["max_drawdown"], performances.values())))],
        "max_position_term": [max(list(map(lambda x: x["max_position_term"], performances.values())))],
        "max_position_size": [max(list(map(lambda x: x["max_position_size"], performances.values())))],
        "average_trade_size": average_trade_size
    }
    print(json.dumps(result))

    if args.upload:
        detail = pandas.DataFrame(result)
        detail.to_csv("settings/simulates.csv", index=None)
        slack.file_post("csv", "settings/simulates.csv", channel="stock_alert")
    return result

def output_setting(args, strategy_setting, score, validate_score, strategy_simulator, report):
    monitor_size = strategy_simulator.strategy_creator.combination_setting.monitor_size
    monitor_size_ratio = monitor_size / report["average_trade_size"]
    with open("settings/%s" % strategy.get_filename(args), "w") as f:
        f.write(json.dumps({
            "date": args.date,
            "term": args.validate_term,
            "score": int(score),
            "validate_score": int(validate_score),
            "monitor_size": monitor_size,
            "monitor_size_ratio": monitor_size_ratio,
            "setting": strategy_setting.__dict__
        }))

def walkforward(args, data, terms, strategy_simulator):
    performances = {}
    # 最適化
    if args.optimize_count > 0 and not args.ignore_optimize:
        d = copy.deepcopy(data)
        strategy_setting, score = strategy_optimize(args, data, terms, strategy_simulator)
        objective(args, strategy_setting, data, terms, strategy_simulator) # 選ばれた戦略スコアを表示するため
        print(strategy_setting.__dict__)
    else:
        _, strategy_setting = strategy.load_strategy_setting(args)
        if args.output:
            print("Need -o parameter or Using --ignore_optimize. don't output simulate setting.")
            exit()

    # 検証
    terms = sorted(validate_terms, key=lambda x: x["start_date"])
    for term in terms:
        print(term)
        start_date = utils.to_format_by_term(term["start_date"], args.tick)
        end_date = utils.to_format_by_term(term["end_date"], args.tick)
        # 検証期間で結果を試す
        d = copy.deepcopy(data)
        result = strategy_simulator.simulates(strategy_setting, d, start_date, end_date, True)

        performances[utils.to_format(utils.to_datetime_by_term(end_date, args.tick))] = result

    # 結果の表示 =============================================================================
    report = create_performance(strategy_simulator.simulator_setting, performances)
    validate_score = -get_score(args, performances.values(), strategy_simulator.simulator_setting, strategy_setting)
    print(validate_score)

    if args.output:
        output_setting(args, strategy_setting, score, validate_score, strategy_simulator, report)


######################################################################
print(utils.timestamp())
proc_start_time = time.time()


if args.assets is None:
    assets = Loader.assets()
    simulate_setting = create_setting(args, assets["assets"])
else:
    simulate_setting = create_setting(args, args.assets)

# 戦略の選択
combination_setting = strategy.create_combination_setting(args)
strategy_creator = strategy.load_strategy_creator(args, combination_setting)

# 翌期間用の設定を出力する
# 都度読み込むと遅いので全部読み込んでおく
terms, validate_terms = create_terms(args)
min_start_date = min(list(map(lambda x: x["start_date"], terms)))
start = utils.to_format_by_term(min_start_date, args.tick)
end = utils.to_datetime(args.date)
if args.tick:
    end = end + datetime.timedelta(hours=9)
end = utils.to_format_by_term(end, args.tick)
strategy_simulator = StrategySimulator(simulate_setting, strategy_creator, verbose=args.verbose)
codes, validate_codes, daterange = strategy_simulator.select_codes(args, start, end)

print("target : %s" % codes)

data = load(args, codes, terms, daterange)

# 期間ごとに最適化
terms = sorted(terms, key=lambda x: x["start_date"])

# 結果ログを削除
params = ["rm", "-rf", "settings/simulate.log"]
subprocess.call(params)

if args.random > 0:
    # 指定回数ランダムで最適化して検証スコアが高い
    params = ["rm", "-rf", "settings/tmp"]
    subprocess.call(params)

    params = ["mkdir", "settings/tmp"]
    subprocess.call(params)

    for i in range(args.random):
        walkforward(args, data, terms, strategy_simulator)
        filename = strategy.get_filename(args)
        params = ["cp", "settings/%s" % (filename), "settings/tmp/%s_%s" % (i, filename)]
        subprocess.call(params)

    params = ["sh", "simulator/copy_highest_score_setting.sh", strategy.get_prefix(args)]
    subprocess.call(params)
else:
    walkforward(args, data, terms, strategy_simulator)

print(utils.timestamp())
proc_end_time = time.time()
print("proc time: %s" % (proc_end_time - proc_start_time))
