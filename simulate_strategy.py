# -*- coding: utf-8 -*-
import os
import sys
import pandas
import json
import time
import numpy
import copy
import subprocess
import datetime
import random
from skopt import gp_minimize
from multiprocessing import Pool
from argparse import ArgumentParser

sys.path.append("lib")
import cache
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
parser.add_argument("--assets", type=int, action="store", default=None, dest="assets", help="assets")
parser.add_argument("-n", "--n_calls", type=int, action="store", default=100, help="simulate n_calls")
parser.add_argument("-j", "--jobs", type=int, action="store", default=8, dest="jobs", help="実行ジョブ数")
parser.add_argument("-v", action="store_true", default=False, dest="verbose", help="debug log")
parser.add_argument("--output", action="store_true", default=False, dest="output", help="設定をファイルに出力")
parser.add_argument("--random", type=int, action="store", default=0, dest="random", help="ランダム学習の回数")
parser.add_argument("--auto_stop_loss", action="store_true", default=False, dest="auto_stop_loss", help="自動損切")
parser.add_argument("--apply_compound_interest", action="store_true", default=False, dest="apply_compound_interest", help="複利を適用")
parser.add_argument("--use_optimized_init", type=int, action="store", default=0, dest="use_optimized_init", help="どこまで初期値に最適化後の設定を使うか")
parser.add_argument("--montecarlo", action="store_true", default=False, dest="montecarlo", help="ランダム取引")
parser = strategy.add_options(parser)
args = parser.parse_args()

# start を1/1, endを12/31にしてしまうと前後のデータがないのでロードに失敗する

def create_cache_name(args):
    prefix = strategy.get_prefix(args)
    params = [args.date, args.validate_term, args.count, args.optimize_count, args.tick]
    params = list(map(lambda x: str(x), params))
    return "%s%s" % (prefix, "_".join(params))

def create_setting(args, assets):
    setting = strategy.create_simulator_setting(args)
    setting.min_data_length = args.validate_term * 10
    setting.assets = assets
    setting.commission = 150
    setting.debug = args.verbose
    setting.auto_stop_loss = args.auto_stop_loss
    return setting

def create_simulator_data(param):
    code = param["code"]
    start_date = param["start_date"]
    end_date = param["end_date"]
    args = param["args"]

    settings = strategy.LoadSettings()
    settings.with_stats = False # startegy_simlator側で各足毎に統計値を出力するのでここでは計算しない

    cacher = cache.Cache("/tmp/simulator")
    name = "_".join([create_cache_name(args), str(code), start_date, end_date])
    if cacher.exists(name):
        data = cacher.get(name)
        print("cache loaded:", code, start_date, end_date)
    else:
        data = strategy.load_simulator_data(code, start_date, end_date, args, settings)
        cacher.create(name, data)

    return data

def load_index(args, start_date, end_date):
    index = {}

    if args.tick:
        return index # 指標はティックデータない

    for k in ["nikkei"]:
        d = Loader.load_index(k, start_date, end_date, with_filter=True, strict=False)
        d = utils.add_stats(d)
        d = utils.add_cs_stats(d)
        index[k] = d

    return index

def load(args, codes, terms, daterange, combination_setting):
    min_start_date = min(list(map(lambda x: x["start_date"], terms)))
    prepare_term = utils.relativeterm(args.validate_term, args.tick)
    start_date = utils.to_format(min_start_date - prepare_term)
    end_date = utils.format(args.date)
    strategy_creator = strategy.load_strategy_creator(args, combination_setting)

    print("loading %s %s %s" % (len(codes), start_date, end_date))
    data = {}
    params = list(map(lambda x: {"code": x, "start_date": utils.to_format(daterange[x][0]), "end_date": utils.to_format(daterange[x][-1]), "args": args}, codes))

    try:
        p = Pool(8)
        ret = p.map(create_simulator_data, params)
        for r in ret:
            if r is None:
                continue
            print("add_data: ", utils.timestamp(), r.code)
            data[r.code] = strategy_creator.add_data(r)
    except KeyboardInterrupt:
        p.close()
        exit()
    finally:
        p.close()

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
    lose = list(filter(lambda x: x < 0, gain))
    drawdown = list(map(lambda x: x["drawdown"], scores))
    max_drawdown = list(map(lambda x: x["max_drawdown"], scores))
    trade = list(map(lambda x: x["trade"], scores))
    win_trade = list(map(lambda x: x["win_trade"], scores))
    profit_factor = sum(win) / abs(sum(lose)) if abs(sum(lose)) > 0 else sum(win)
    gain_per_trade = sum(gain) / sum(trade) if sum(trade) > 0 else 0

    return {
        "gain": gain,
        "win": win,
        "lose": lose,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "gain_per_trade": gain_per_trade,
        "trade": trade,
        "win_trade": win_trade
    }

def print_score_stats(name, score, score_stats, assets, strategy_setting):
    stats = [
        "max_dd", round(max(score_stats["max_drawdown"]), 2),
        "min:", round(min(score_stats["gain"]) / assets, 2),
        "max:", round(max(score_stats["gain"]) / assets, 2),
        "sum:", sum(score_stats["gain"]),
        "win:", sum(score_stats["win"]),
        "lose:", sum(score_stats["lose"]),
        "pf:", round(score_stats["profit_factor"], 2),
        "gpt:", round(score_stats["gain_per_trade"], 2),
        "t:", sum(score_stats["trade"]),
        "wt:", sum(score_stats["win_trade"])]

    print(utils.timestamp(), name, stats, score)
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
#        score_stats["profit_factor"] < 1.5, # プロフィットファクター（総純利益 / 総損失）が1.5以下
        score_stats["profit_factor"] < 1.1, # プロフィットファクター（総純利益 / 総損失）が1.5以下
#        gain_per_trade < 5000, # 1トレードあたりの平均利益が5000円以下
    ]

    score = score_stats["profit_factor"] * sum(score_stats["gain"]) * (1 - max(score_stats["max_drawdown"])) * sum(score_stats["win_trade"])
    if any(ignore):
        score = 0

    print_score_stats("tick:", score, score_stats, simulator_setting.assets, strategy_setting)

    return score

def simulate_by_term(param):
    strategy_simulator = param[0]
    return strategy_simulator.simulates(*param[1:])

def select_data(codes, stocks):
    select = {"data": {}, "index": stocks["index"], "args": stocks["args"]}

    for code in codes:
        select["data"][code] = stocks["data"][code]

    return select

# 1つの設定でstart~endまでのterm毎のシミュレーション結果を返す
def simulate_by_multiple_term(strategy_setting, stocks, terms, strategy_simulator):
    tick = stocks["args"].tick
    params = []
    strategy_simulator.simulator_setting.strategy = None
    for term in terms:
        # term毎にデータを分けてシミュレートした結果の平均をスコアとして返す
        start = utils.to_format_by_term(term["start_date"], tick)
        end = utils.to_format_by_term(term["end_date"], tick)
        codes, _, _ = strategy_simulator.select_codes(stocks["args"], start, end)
        select = select_data(codes, stocks)
        params.append((strategy_simulator, strategy_setting, select, start, end))

    try:
        p = Pool(int(stocks["args"].jobs))
        scores = p.map(simulate_by_term, params)
    except KeyboardInterrupt:
        p.close()
        exit()
    finally:
        p.close()
    return scores

# パラメータ評価用の関数
# 指定戦略を全銘柄に対して最適化
def objective(args, strategy_setting, stocks, terms, strategy_simulator):
    print(strategy_setting.__dict__)
    try:
        scores = simulate_by_multiple_term(strategy_setting, stocks, terms, strategy_simulator)
        score = get_score(args, scores, strategy_simulator.simulator_setting, strategy_setting)
    except Exception as e:
        print("skip objective. %s" % e)
        import traceback
        traceback.print_exc()
        score = 0
    return -score

def strategy_optimize(args, stocks, terms, strategy_simulator):
    print("strategy_optimize: %s" % (utils.timestamp()))
    strategy_setting = strategy.StrategySetting()

    # 現在の期間で最適な戦略を選択
    space = strategy_simulator.strategy_creator(args).ranges()
    n_random_starts = int(args.n_calls/10) if args.random > 0 else 10
    random_state = int(time.time()) if args.random > 0 else None
    res_gp = gp_minimize(
        lambda x: objective(args, strategy_setting.by_array(x), stocks, terms, strategy_simulator),
        space, n_calls=args.n_calls, n_random_starts=n_random_starts, random_state=random_state)
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
    # レポート出力
    with open("settings/performance.json", "w") as f:
        f.write(json.dumps(performances))

    # 簡易レポート
    for k, v in sorted(performances.items(), key=lambda x: utils.to_datetime(x[0])):
        print(k, v)

    gain = sum(list(map(lambda x: x["gain"], performances.values())))
    average_trade_size = numpy.average(list(map(lambda x: len(x["codes"]), performances.values())))

    result = {
        "gain": gain,
        "return": round(gain / simulator_setting.assets, 3),
        "max_drawdown": max(list(map(lambda x: x["max_drawdown"], performances.values()))),
        "max_position_term": max(list(map(lambda x: x["max_position_term"], performances.values()))),
        "max_position_size": max(list(map(lambda x: x["max_position_size"], performances.values()))),
        "average_trade_size": round(average_trade_size),
        "max_unavailable_assets": max(list(map(lambda x: x["max_unavailable_assets"], performances.values()))),
        "trade": sum(list(map(lambda x: x["trade"], performances.values()))),
        "win_trade": sum(list(map(lambda x: x["win_trade"], performances.values()))),
    }
    print(json.dumps(result))

    return result

def output_setting(args, strategy_settings, score, validate_score, strategy_simulator, report):
    monitor_size = strategy_simulator.combination_setting.monitor_size
    monitor_size_ratio = (monitor_size / report["average_trade_size"]) if report["average_trade_size"] > 0 else 1
    filename = "simulate_settings/%s" % strategy.get_filename(args)
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, "w") as f:
        f.write(json.dumps({
            "date": args.date,
            "term": args.validate_term,
            "score": int(score) * int(validate_score) * -1,
            "optimize_score": int(score),
            "validate_score": int(validate_score),
            "monitor_size": monitor_size,
            "monitor_size_ratio": monitor_size_ratio,
            "max_position_size": strategy_simulator.combination_setting.max_position_size,
            "position_sizing": strategy_simulator.combination_setting.position_sizing,
            "stop_loss_rate": strategy_simulator.simulator_setting.stop_loss_rate,
            "taking_rate": strategy_simulator.simulator_setting.taking_rate,
            "setting": list(map(lambda x: x.__dict__, strategy_settings)),
            "seed": strategy_simulator.combination_setting.seed,
            "report": report,
        }))

def walkforward(args, data, terms, strategy_simulator, combination_setting):
    performances = {}
    # 最適化
    if args.optimize_count > 0 and not args.ignore_optimize:
        if args.use_optimized_init == 0:
            strategy_simulator.combination_setting.seed = [time.time()]
        else:
            strategy_simulator.combination_setting.seed = combination_setting.seed[:args.use_optimized_init] + [time.time()]

        d = copy.deepcopy(data)
        strategy_setting, score = strategy_optimize(args, data, terms, strategy_simulator)
        objective(args, strategy_setting, data, terms, strategy_simulator) # 選ばれた戦略スコアを表示するため
        strategy_settings = strategy_simulator.strategy_settings[:args.use_optimized_init] + [strategy_setting]
        print(strategy_setting.__dict__)
    else:
        _, strategy_settings = strategy.load_strategy_setting(args)
        if args.output:
            print("Need -o parameter or Using --ignore_optimize. don't output simulate setting.")
            exit()

    # 検証
    terms = sorted(validate_terms, key=lambda x: x["start_date"])
    for term in terms:
        start_date = utils.to_format_by_term(term["start_date"], args.tick)
        end_date = utils.to_format_by_term(term["end_date"], args.tick)
        # 検証期間で結果を試す
        d = copy.deepcopy(data)
        result = strategy_simulator.simulates(strategy_settings[-1], d, start_date, end_date)

        if args.apply_compound_interest: # 複利を適用
            strategy_simulator.simulator_setting.assets += result["gain"]
            print("assets:", strategy_simulator.simulator_setting.assets, result["gain"])

        performances[utils.to_format(utils.to_datetime_by_term(end_date, args.tick))] = result

    # 検証スコア
    validate_score = -get_score(args, performances.values(), strategy_simulator.simulator_setting, strategy_settings[-1])
    print(validate_score)

    # 結果の表示 =============================================================================
    report = create_performance(strategy_simulator.simulator_setting, performances)

    if args.output:
        print("strategy_setting:", len(strategy_settings))
        output_setting(args, strategy_settings, score, validate_score, strategy_simulator, report)


######################################################################
print(utils.timestamp())
proc_start_time = time.time()

if args.assets is None:
    assets = Loader.assets()
    simulate_setting = create_setting(args, assets["assets"])
else:
    simulate_setting = create_setting(args, args.assets)

if args.optimize_count > 0 and not args.ignore_optimize and args.use_optimized_init == 0:
    combination_setting = strategy.create_combination_setting(args, use_json=False)
    strategy_settings = []
else:
    combination_setting = strategy.create_combination_setting(args)
    _, strategy_settings = strategy.load_strategy_setting(args)

combination_setting.montecarlo = args.montecarlo

strategy_simulator = StrategySimulator(simulate_setting, combination_setting, strategy_settings, verbose=args.verbose)

# 翌期間用の設定を出力する
# 都度読み込むと遅いので全部読み込んでおく
terms, validate_terms = create_terms(args)
min_start_date = min(list(map(lambda x: x["start_date"], terms)))
start = utils.to_format_by_term(min_start_date, args.tick)
end = utils.to_datetime(args.date)
if args.tick:
    end = end + datetime.timedelta(hours=9)
end = utils.to_format_by_term(end, args.tick)
codes, validate_codes, daterange = strategy_simulator.select_codes(args, start, end)

print("target : %s" % codes)

data = load(args, codes, terms, daterange, combination_setting)

# 期間ごとに最適化
terms = sorted(terms, key=lambda x: x["start_date"])

# 結果ログを削除
params = ["rm", "-rf", "settings/simulate.log"]
subprocess.call(params)

if args.random > 0:
    # 指定回数ランダムで最適化して検証スコアが高い
    params = ["rm", "-rf", "simulate_settings/tmp"]
    subprocess.call(params)

    params = ["mkdir", "simulate_settings/tmp"]
    subprocess.call(params)

    filename = strategy.get_filename(args)
    params = ["cp", "simulate_settings/%s" % (filename), "simulate_settings/tmp/default_%s" % filename]
    subprocess.call(params)

    for i in range(args.random):
        walkforward(args, data, terms, strategy_simulator, combination_setting)

        params = ["cp", "simulate_settings/%s" % (filename), "simulate_settings/tmp/%s_%s" % (i, filename)]
        subprocess.call(params)

    params = ["sh", "simulator/copy_highest_score_setting.sh", strategy.get_prefix(args)]
    subprocess.call(params)
else:
    walkforward(args, data, terms, strategy_simulator, combination_setting)

print(utils.timestamp())
proc_end_time = time.time()
print("proc time: %s" % (proc_end_time - proc_start_time))
