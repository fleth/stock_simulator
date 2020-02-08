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
from loader import Loader, Bitcoin
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
parser.add_argument("-o", type=int, action="store", default=0, dest="optimize_count", help="最適化期間の数")
parser.add_argument("-c", type=int, action="store", default=1, dest="count", help="検証期間の数")
parser.add_argument("-n", "--n_calls", type=int, action="store", default=100, help="simulate n_calls")
parser.add_argument("-j", "--jobs", type=int, action="store", default=8, dest="jobs", help="実行ジョブ数")
parser.add_argument("-v", action="store_true", default=False, dest="verbose", help="debug log")
parser.add_argument("--assets", type=int, action="store", default=None, dest="assets", help="assets")
parser.add_argument("--commission", type=int, action="store", default=150, dest="commission", help="commission")
parser.add_argument("--output", action="store_true", default=False, dest="output", help="設定をファイルに出力")
parser.add_argument("--random", type=int, action="store", default=0, dest="random", help="ランダム学習の回数")
parser.add_argument("--skip_optimized", action="store_true", default=False, dest="skip_optimized", help="最適化済みなら最適化をスキップ")
parser.add_argument("--ignore_optimize", action="store_true", default=False, dest="ignore_optimize", help="最適化を実施しない")
parser.add_argument("--use_optimized_init", type=int, action="store", default=0, dest="use_optimized_init", help="どこまで初期値に最適化後の設定を使うか")
parser.add_argument("--output_dir", type=str, action="store", default="simulate_settings", dest="output_dir", help="")
parser.add_argument("--apply_compound_interest", action="store_true", default=False, dest="apply_compound_interest", help="複利を適用")
parser.add_argument("--montecarlo", action="store_true", default=False, dest="montecarlo", help="ランダム取引")
parser.add_argument("--use_cache", action="store_true", default=False, dest="use_cache", help="キャッシュを使う")
parser.add_argument("--performance", action="store_true", default=False, dest="performance", help="パフォーマンスレポートを出力する")
parser = strategy.add_options(parser)
args = parser.parse_args()

# !!! for 使ってるので遅い !!!
def to_jsonizable(dic):
    jsonizable = {}
    for k, v in dic.items():
        if type(v) is dict:
            jsonizable[k] = to_jsonizable(v)
        elif "numpy" in str(type(v)):
            jsonizable[k] = numpy.array([v]).tolist()[0]
        else:
            jsonizable[k] = v
    return jsonizable

# start を1/1, endを12/31にしてしまうと前後のデータがないのでロードに失敗する
def create_cache_name(args):
    prefix = strategy.get_prefix(args)
    params = [args.date, args.validate_term, args.count, args.optimize_count, args.daytrade]
    params = list(map(lambda x: str(x), params))
    return "%s%s" % (prefix, "_".join(params))

def create_setting(args, assets):
    setting = strategy.create_simulator_setting(args)
    setting.min_data_length = args.validate_term * 10
    setting.assets = assets
    setting.commission = args.commission
    setting.debug = args.verbose
    return setting

def create_simulator_data(param):
    code = param["code"]
    start_date = param["start_date"]
    end_date = param["end_date"]
    args = param["args"]

    cacher = cache.Cache("/tmp/simulator")
    name = "_".join([create_cache_name(args), str(code), start_date, end_date])
    if cacher.exists(name) and args.use_cache:
        data = cacher.get(name)
        print("cache loaded:", code, start_date, end_date)
    else:
        data = strategy.load_simulator_data(code, start_date, end_date, args)
        cacher.create(name, data)

    return data

def load_index(args, start_date, end_date):
    index = {}

    if args.daytrade:
        return index # 指標はティックデータない

    for k in ["nikkei"]:
        d = Loader.load_index(k, start_date, end_date, with_filter=True, strict=False)
        d = utils.add_stats(d)
        d = utils.add_cs_stats(d)
        index[k] = d

    return index

def load(args, codes, terms, daterange, combination_setting):
    min_start_date = min(list(map(lambda x: x["start_date"], terms)))
    prepare_term = utils.relativeterm(args.validate_term, args.daytrade)
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
    if args.daytrade:
        score = get_daytrade_score(scores, simulator_setting, strategy_setting)
    else:
        score = get_default_score(scores, simulator_setting, strategy_setting)
    return score

def sorted_values(scores, key):
    sorted_scroes = sorted(scores, key=lambda x:utils.to_datetime(x["start_date"]))
    values = list(map(lambda x: x[key], sorted_scroes))
    return values

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
        "win_trade": win_trade,
        "with_weight": get_score_stats_with_weight(scores),
    }

def get_score_stats_with_weight(scores):
    stats = {}

    term = len(scores)
    # with weight
    weight = 1.2 / term 
    weights = numpy.array(list(range(term))) * weight
    weights = numpy.power(weights, numpy.array([3] * len(weights)))
    gains = numpy.array(sorted_values(scores, "gain")) * numpy.array(weights) # 過去の結果ほど重要度を下げ

    profits = list(filter(lambda x: x > 0, gains))
    loss = list(filter(lambda x: x < 0, gains))

    stats["term"] = term
    stats["min_loss"] = min(loss) if len(loss) > 0 else 0
    stats["gain"] = sum(gains)
    stats["max_gain"] = max(profits) if len(profits) > 0 else 0
    stats["average_gain"] = numpy.average(profits).item() if len(profits) > 0 else 0
    stats["median_gain"] = numpy.median(profits).item() if len(profits) > 0 else 0
    stats["std_gain"] = numpy.std(profits).item() if len(profits) > 0 else 0
    stats["profits"] = sum(profits) if len(profits) > 0 else 0
    stats["loss"] = sum(loss) if len(loss) > 0 else 0

    stats["win"] = len(profits)
    stats["lose"] = len(loss)

    stats["profit_factor"] = stats["profits"] / (abs(stats["loss"]) if stats["loss"] < 0 else 1)

    stats["max_impact"] = float(stats["max_gain"] / sum(profits)) if stats["gain"] > 0 else 0
    return stats

def print_score_stats(name, score, score_with_weights, score_stats, assets, strategy_setting):
    stats = [
        "max_dd", "{:.2f}".format(max(score_stats["max_drawdown"])),
        "min:", "{:.2f}".format(min(score_stats["gain"]) / assets),
        "max:", "{:.2f}".format(max(score_stats["gain"]) / assets),
        "sum:", "{:.2f}".format(sum(score_stats["gain"])),
        "win:", "{:.2f}".format(sum(score_stats["win"])),
        "lose:", "{:.2f}".format(sum(score_stats["lose"])),
        "pf:", "{:.2f}".format(score_stats["profit_factor"]),
        "gpt:", "{:.2f}".format(score_stats["gain_per_trade"]),
        "t:", sum(score_stats["trade"]),
        "wt:", sum(score_stats["win_trade"])]

    print(utils.timestamp(), name, stats, "{:.2f}".format(score), "{:.2f}".format(score_with_weights))
    setting = {"name": name, "stats": stats, "score": score, "setting": strategy_setting.to_dict()}
    with open("settings/simulate.log", "a") as f:
        f.write(json.dumps(to_jsonizable(setting)))
        f.write("\n")

def get_default_score(scores, simulator_setting, strategy_setting):
    score_stats = get_score_stats(scores)

    ignore = [
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
        sum(score_stats["trade"]) < 30, # 取引数が少ない
        score_stats["profit_factor"] < 1.5, # プロフィットファクター（総純利益 / 総損失）が1.5以下
    ]

    if any(ignore):
        score = 0
        score_with_weights = 0
    else:
        score = score_stats["profit_factor"] * sum(score_stats["gain"]) * (1 - max(score_stats["max_drawdown"]))
        score_with_weights = get_score_with_weight(score_stats["with_weight"])
        score = score * score_with_weights

    print_score_stats("default:", score, score_with_weights, score_stats, simulator_setting.assets, strategy_setting)

    return score

def get_daytrade_score(scores, simulator_setting, strategy_setting):
    score_stats = get_score_stats(scores)

    ignore = [
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
        score_stats["profit_factor"] < 1.1, # プロフィットファクター（総純利益 / 総損失）が1.5以下
    ]

    if any(ignore):
        score = 0
        score_with_weights = 0
    else:
        score = score_stats["profit_factor"] * sum(score_stats["gain"]) * (1 - max(score_stats["max_drawdown"])) * sum(score_stats["win_trade"])
        score_with_weights = get_score_with_weight(score_stats["with_weight"])
        score = score * score_with_weights

    print_score_stats("daytrade:", score, score_with_weights, score_stats, simulator_setting.assets, strategy_setting)

    return score

def get_score_with_weight(stats):
    ignore_conditions = [
        stats["gain"] <= 0,
        stats["std_gain"] == 0,
        stats["max_impact"] > 0.3, # 総利益の30%以上を一か月で得ている場合除外
    ]

    if any(ignore_conditions):
        score = 0
    else:
        win_rate = (stats["win"] / stats["term"]) # 勝率が高い
        impact_per_trade = (1 - stats["max_impact"]) # 特定のトレードの重要度が低い

        score = (stats["average_gain"] * stats["median_gain"]) / stats["std_gain"]

        score = score * stats["profit_factor"] * win_rate * impact_per_trade
    return score

def simulate_by_term(param):
    strategy_simulator = param[0]
    return strategy_simulator.simulates(*param[1:])

def select_data(codes, stocks, start, end):
    select = {"data": {}, "index": stocks["index"], "args": stocks["args"]}

    args = select["args"]

    for code in codes:
        if not code in stocks["data"].keys():
            continue
        start_date = utils.to_format(utils.to_datetime_by_term(start,args.daytrade) - utils.relativeterm(1, args.daytrade))
        select["data"][code] = stocks["data"][code].split(start_date, end)

    return select

def simulate_params(stocks, terms, strategy_simulator):
    daytrade = stocks["args"].daytrade
    params = []
    strategy_simulator.simulator_setting.strategy = None
    for term in terms:
        # term毎にデータを分けてシミュレートした結果の平均をスコアとして返す
        start = utils.to_format_by_term(term["start_date"], daytrade)
        end = utils.to_format_by_term(term["end_date"], daytrade)
        codes, _, _ = strategy_simulator.select_codes(stocks["args"], start, end)
        select = select_data(codes, stocks, start, end)

        params.append((select, start, end))
    return params

# 1つの設定でstart~endまでのterm毎のシミュレーション結果を返す
def simulate_by_multiple_term(stocks, params):
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
def objective(args, strategy_setting, stocks, params, validate_params, strategy_simulator):
    print(strategy_setting.__dict__)
    try:
        params = list(map(lambda x: (strategy_simulator, strategy_setting) + x, params))
        scores = simulate_by_multiple_term(stocks, params)
        score = get_score(args, scores, strategy_simulator.simulator_setting, strategy_setting)
        if score > 0:
            validate_params = list(map(lambda x: (strategy_simulator, strategy_setting) + x, validate_params))
            validate_scores = simulate_by_multiple_term(stocks, validate_params)
            validate_score = get_score(args, validate_scores, strategy_simulator.simulator_setting, strategy_setting)
            score = score * validate_score
    except Exception as e:
        print("skip objective. %s" % e)
        import traceback
        traceback.print_exc()
        score = 0
    return -score

def strategy_optimize(args, stocks, params, validate_params, strategy_simulator):
    print("strategy_optimize: %s" % (utils.timestamp()))
    strategy_setting = strategy.StrategySetting()

    # 現在の期間で最適な戦略を選択
    space = strategy_simulator.strategy_creator(args).ranges()
    n_random_starts = int(args.n_calls/10) if args.random > 0 else 10
    random_state = int(time.time()) if args.random > 0 else None
    res_gp = gp_minimize(
        lambda x: objective(args, strategy_setting.by_array(x), stocks, params, validate_params, strategy_simulator),
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
        end_date = valid_end_date - utils.relativeterm(args.validate_term, args.daytrade)
        start_date = end_date - utils.relativeterm(args.validate_term*args.optimize_count, args.daytrade)
        term = {"start_date": start_date, "end_date": end_date}
        validate_term = {"start_date": end_date, "end_date": valid_end_date}
        term, validate_term = term_filter(args, term, validate_term)
        terms.append(term)
        validate_terms.append(validate_term)
        valid_end_date = start_date
    print(list(map(lambda x: "%s - %s" % (str(x["start_date"]), str(x["end_date"])), terms)))
    print(list(map(lambda x: "%s - %s" % (str(x["start_date"]), str(x["end_date"])), validate_terms)))

    terms = sorted(terms, key=lambda x: x["start_date"])
    validate_terms = sorted(validate_terms, key=lambda x: x["start_date"])

    return terms, validate_terms

def term_filter(args, term, validate_term):
    if not args.daytrade:
        return term, validate_term

    if args.code in Bitcoin().exchanges:
        term["start_date"] += datetime.timedelta(days=1)
        term["end_date"] += datetime.timedelta(days=1)
        validate_term["start_date"] += datetime.timedelta(days=1)
        validate_term["end_date"] += datetime.timedelta(days=1)
        return term, validate_term

    term["start_date"] += datetime.timedelta(days=1)
    term["start_date"] += datetime.timedelta(hours=9)
    term["end_date"] += datetime.timedelta(hours=15)
    validate_term["start_date"] += datetime.timedelta(days=1)
    validate_term["start_date"] += datetime.timedelta(hours=9)
    validate_term["end_date"] += datetime.timedelta(hours=15)
    return term, validate_term

def create_performance(args, simulator_setting, performances):
    # レポート出力
    if args.performance:
        filename = "%s/performances/%sperformance.json" % (args.output_dir, strategy.get_prefix(args))
        with open(filename, "w") as f:
            f.write(json.dumps(to_jsonizable(performances)))

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
        "sum_contract_price": sum(list(map(lambda x: x["sum_contract_price"], performances.values()))),
        "trade": sum(list(map(lambda x: x["trade"], performances.values()))),
        "win_trade": sum(list(map(lambda x: x["win_trade"], performances.values()))),
    }
    print(json.dumps(to_jsonizable(result)))

    return result

def output_setting(args, strategy_settings, strategy_simulator, score, optimize_score, validate_score, optimize_report, validate_report):
    filename = "%s/%s" % (args.output_dir, strategy.get_filename(args))
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, "w") as f:
        f.write(json.dumps(to_jsonizable({
            "date": args.date,
            "term": args.validate_term,
            "score": int(score),
            "optimize_score": int(optimize_score),
            "validate_score": int(validate_score),
            "max_position_size": strategy_simulator.combination_setting.max_position_size,
            "position_sizing": strategy_simulator.combination_setting.position_sizing,
            "stop_loss_rate": strategy_simulator.simulator_setting.stop_loss_rate,
            "taking_rate": strategy_simulator.simulator_setting.taking_rate,
            "min_unit": strategy_simulator.simulator_setting.min_unit,
            "setting": list(map(lambda x: x.__dict__, strategy_settings)),
            "seed": strategy_simulator.combination_setting.seed,
            "optimize_report": optimize_report,
            "validate_report": validate_report,
            "use_limit": args.use_limit
        })))

def validation(args, stocks, terms, strategy_simulator, combination_setting, strategy_settings):
    performances = {}
    strategy_simulator.strategy_settings = strategy_settings
    strategy_simulator.combination_setting = combination_setting
    params = simulate_params(stocks, terms, strategy_simulator)
    for param in params:
        _, start_date, end_date = param
        result = simulate_by_term((strategy_simulator, strategy_settings[-1]) + param)

        if args.apply_compound_interest: # 複利を適用
            strategy_simulator.simulator_setting.assets += result["gain"]
            print("assets:", strategy_simulator.simulator_setting.assets, result["gain"])

        performances[utils.to_format(utils.to_datetime_by_term(end_date, args.daytrade))] = result

    # 検証スコア
    score = -get_score(args, performances.values(), strategy_simulator.simulator_setting, strategy_settings[-1])

    # 結果の表示 =============================================================================
    report = create_performance(args, strategy_simulator.simulator_setting, performances)

    return score, report

def walkforward(args, stocks, terms, validate_terms, strategy_simulator, combination_setting):
    # 最適化
    if args.optimize_count > 0 and not args.ignore_optimize:
        strategy_simulator.combination_setting = combination_setting

        if args.use_optimized_init == 0:
            strategy_simulator.combination_setting.seed = [time.time()]
        else:
            strategy_simulator.combination_setting.seed = combination_setting.seed[:args.use_optimized_init] + [time.time()]

        params = simulate_params(stocks, terms, strategy_simulator)
        validate_params = simulate_params(stocks, validate_terms, strategy_simulator)
        strategy_setting, score = strategy_optimize(args, stocks, params, validate_params, strategy_simulator)
        objective(args, strategy_setting, stocks, params, validate_params, strategy_simulator) # 選ばれた戦略スコアを表示するため
        strategy_settings = strategy_simulator.strategy_settings[:args.use_optimized_init] + [strategy_setting]
        print(strategy_setting.__dict__)
    else:
        _, strategy_settings = strategy.load_strategy_setting(args)
        if args.output:
            print("Need -o parameter or Using --ignore_optimize. don't output simulate setting.")
            exit()

    # 検証
    validate_combination_setting = copy.copy(combination_setting)
    validate_combination_setting.use_limit = False
    optimize_score, optimize_report = validation(args, stocks, terms, strategy_simulator, validate_combination_setting, strategy_settings)
    validate_score, validate_report = validation(args, stocks, validate_terms, strategy_simulator, validate_combination_setting, strategy_settings)
    print(validate_score)

    if args.output:
        print("strategy_setting:", len(strategy_settings))
        output_setting(args, strategy_settings, strategy_simulator, score, optimize_score, validate_score, optimize_report, validate_report)


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
start = utils.to_format_by_term(min_start_date, args.daytrade)
end = utils.to_datetime(args.date)
if args.daytrade:
    end += datetime.timedelta(hours=15)
end = utils.to_format_by_term(end, args.daytrade)
codes, validate_codes, daterange = strategy_simulator.select_codes(args, start, end)

print("target : %s" % codes, start, end)

stocks = load(args, codes, terms, daterange, combination_setting)

# 期間ごとに最適化
terms = sorted(terms, key=lambda x: x["start_date"])

# 結果ログを削除
params = ["rm", "-rf", "settings/simulate.log"]
subprocess.call(params)

params = ["mkdir", "-p", "%s/performances" % args.output_dir]
subprocess.call(params)

if args.random > 0:
    # 指定回数ランダムで最適化して検証スコアが高い
    params = ["rm", "-rf", "%s/tmp" % args.output_dir]
    subprocess.call(params)

    params = ["mkdir", "%s/tmp" % args.output_dir]
    subprocess.call(params)

    filename = strategy.get_filename(args)
    params = ["cp", "%s/%s" % (args.output_dir, filename), "%s/tmp/default_%s" % (args.output_dir, filename)]
    status = subprocess.call(params)

    if status == 0 and args.skip_optimized:
        print("skip. optimized.")
        exit()

    for i in range(args.random):
        walkforward(args, stocks, terms, validate_terms, strategy_simulator, combination_setting)

        params = ["cp", "%s/%s" % (args.output_dir, filename), "%s/tmp/%s_%s" % (args.output_dir,i, filename)]
        subprocess.call(params)

    params = ["sh", "simulator/copy_highest_score_setting.sh", strategy.get_prefix(args), args.output_dir]
    subprocess.call(params)
else:
    walkforward(args, stocks, terms, validate_terms, strategy_simulator, combination_setting)

print(utils.timestamp())
proc_end_time = time.time()
print("proc time: %s" % (proc_end_time - proc_start_time))
