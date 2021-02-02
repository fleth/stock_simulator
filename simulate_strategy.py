# -*- coding: utf-8 -*-
import os
import sys
import math
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
import utils
import slack
import strategy
from loader import Loader, Bitcoin
from simulator import Simulator, SimulatorSetting, SimulatorData
from strategy_simulator import StrategySimulator

# /skopt/learning/gaussian_process/gpr.py:294: FutureWarning: Beginning in version 0.22, arrays of bytes/strings will be converted to decimal numbers if dtype='numeric'. It is recommended that you convert the array to a float dtype before using it in scikit-learn, for example by using your_array = your_array.astype(np.float64).
import warnings
warnings.simplefilter('ignore', FutureWarning)

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

default_output_dir = "simulate_settings"
parser = ArgumentParser()
parser.add_argument("date", type=str)
parser.add_argument("validate_term", type=int) # 最適化期間の10~20%
parser.add_argument("-o", type=int, action="store", default=0, dest="optimize_count", help="最適化期間の数")
parser.add_argument("-c", type=int, action="store", default=1, dest="count", help="検証期間の数")
parser.add_argument("-n", "--n_calls", type=int, action="store", default=100, help="simulate n_calls")
parser.add_argument("-j", "--jobs", type=int, action="store", default=8, dest="jobs", help="実行ジョブ数")
parser.add_argument("-v", action="store_true", default=False, dest="verbose", help="debug log")
parser.add_argument("--output", action="store_true", default=False, dest="output", help="設定をファイルに出力")
parser.add_argument("--random", type=int, action="store", default=0, dest="random", help="ランダム学習の回数")
parser.add_argument("--skip_optimized", action="store_true", default=False, dest="skip_optimized", help="最適化済みなら最適化をスキップ")
parser.add_argument("--ignore_optimize", action="store_true", default=False, dest="ignore_optimize", help="最適化を実施しない")
parser.add_argument("--use_optimized_init", type=int, action="store", default=0, dest="use_optimized_init", help="どこまで初期値に最適化後の設定を使うか")
parser.add_argument("--output_dir", type=str, action="store", default=default_output_dir, dest="output_dir", help="")
parser.add_argument("--apply_compound_interest", action="store_true", default=False, dest="apply_compound_interest", help="複利を適用")
parser.add_argument("--montecarlo", action="store_true", default=False, dest="montecarlo", help="ランダム取引")
parser.add_argument("--performance", action="store_true", default=False, dest="performance", help="パフォーマンスレポートを出力する")
parser.add_argument("--with_weights", action="store_true", default=False, dest="with_weights", help="重みを引き継ぐ")
parser.add_argument("--amount", type=int, action="store", default=100, dest="amount", help="重みの増加量")
parser.add_argument("--ignore_default", action="store_true", default=False, dest="ignore_default", help="ignore_default")
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

def create_setting(args):
    setting = strategy.create_simulator_setting(args, args.optimize_count == 0)
    setting.debug = args.verbose
    return setting

def create_simulator_data(param):
    code = param["code"]
    start_date = param["start_date"]
    end_date = param["end_date"]
    args = param["args"]

    data = strategy.load_simulator_data(code, start_date, end_date, args)

    return data

def load(args, codes, terms, combination_setting):
    min_start_date = min(list(map(lambda x: x["start_date"], terms)))
    prepare_term = utils.relativeterm(args.validate_term)
    start_date = utils.to_format(min_start_date - prepare_term)
    end_date = utils.format(args.date)
    strategy_creator = strategy.load_strategy_creator(args, combination_setting)

    print("loading %s %s %s" % (len(codes), start_date, end_date))
    data = {}
    params = list(map(lambda x: {"code": x, "start_date": start_date, "end_date": end_date, "args": args}, codes))

    try:
        p = Pool(8)
        ret = p.map(create_simulator_data, params)
        for r in ret:
            if r is None:
                continue
#            print("add_data: ", utils.timestamp(), r.code)
            data[r.code] = strategy_creator.add_data(r)
    except KeyboardInterrupt:
        p.close()
        exit()
    finally:
        p.close()

    index = strategy.load_index(args, start_date, end_date)

    print("loading done")
    return {"data": data, "index": index, "args": args}

def get_score(args, performances, simulator_setting, strategy_setting):
    if args.short:
        score = get_short_score(performances, simulator_setting, strategy_setting)
    elif args.instant:
        score = get_instant_score(performances, simulator_setting, strategy_setting)
    else:
        score = get_default_score(performances, simulator_setting, strategy_setting)
    return score

def sorted_values(performances, key):
    sorted_scroes = sorted(performances, key=lambda x:utils.to_datetime(x["start_date"]))
    values = list(map(lambda x: x[key], sorted_scroes))
    return values

def get_score_stats(performances):
    term = len(performances)
    gain = list(map(lambda x: x["gain"], performances))
    win = list(filter(lambda x: x > 0, gain))
    lose = list(filter(lambda x: x < 0, gain))
    drawdown = list(map(lambda x: x["drawdown"], performances))
    max_drawdown = list(map(lambda x: x["max_drawdown"], performances))
    trade = list(map(lambda x: x["trade"], performances))
    win_trade = list(map(lambda x: x["win_trade"], performances))
    profit_factor = sum(win) / abs(sum(lose)) if abs(sum(lose)) > 0 else 1
    gain_per_trade = sum(gain) / sum(trade) if sum(trade) > 0 else 0
    crash = list(map(lambda x: x["crash"], performances))

    return {
        "term": term,
        "gain": gain,
        "win": win,
        "lose": lose,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "crash": crash,
        "gain_per_trade": gain_per_trade,
        "trade": trade,
        "win_trade": win_trade,
    }

def print_score_stats(name, score, score_stats, assets, strategy_setting):
    stats = [
        "max_dd", "{:.2f}".format(max(score_stats["max_drawdown"])),
        "min:", "{:.2f}".format(min(score_stats["gain"]) / assets),
        "max:", "{:.2f}".format(max(score_stats["gain"]) / assets),
        "sum:", "{:.2f}".format(sum(score_stats["gain"])),
        "win:", "{:.2f}".format(sum(score_stats["win"])),
        "lose:", "{:.2f}".format(sum(score_stats["lose"])),
        "t:", sum(score_stats["trade"]),
        "wt:", sum(score_stats["win_trade"])]

    print(utils.timestamp(), name, stats, "{:.2f}".format(score))
    setting = {"name": name, "stats": stats, "score": score, "setting": strategy_setting.to_dict()}
    with open("settings/simulate.log", "a") as f:
        f.write(json.dumps(to_jsonizable(setting)))
        f.write("\n")

def get_default_score(performances, simulator_setting, strategy_setting):
    if len(performances) == 0:
        return 0

    score_stats = get_score_stats(performances)

    ignore = [
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
        sum(score_stats["trade"]) < score_stats["term"] / 2, # 取引数が少ない
        score_stats["profit_factor"] < 1.0, # プロフィットファクター（総純利益 / 総損失）が1.1以下
    ]

    if any(ignore):
        score = 0
    else:
        score = sum(score_stats["trade"]) * (sum(score_stats["gain"]) / 10000) * (1 - max(score_stats["max_drawdown"]))
        score = score * (score_stats["gain_per_trade"] / 10000)
        score = score * (1 - abs(min(score_stats["crash"])) / simulator_setting.assets)

    print_score_stats("", score, score_stats, simulator_setting.assets, strategy_setting)

    return score

def get_instant_score(performances, simulator_setting, strategy_setting):
    if len(performances) == 0:
        return 0

    score_stats = get_score_stats(performances)

    ignore = [
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
#        sum(score_stats["trade"]) < score_stats["term"] / 2, # 取引数が少ない
        score_stats["profit_factor"] < 1.0, # プロフィットファクター（総純利益 / 総損失）が1.1以下
    ]

    if any(ignore):
        score = 0
    else:
        score = (sum(score_stats["win_trade"]) / sum(score_stats["trade"])) * (sum(score_stats["gain"]) / 10000) * (1 - max(score_stats["max_drawdown"]))

    print_score_stats("instant", score, score_stats, simulator_setting.assets, strategy_setting)

    return score

def get_short_score(performances, simulator_setting, strategy_setting):
    if len(performances) == 0:
        return 0

    score_stats = get_score_stats(performances)

    ignore = [
        sum(score_stats["gain"]) <= 0, # 損益がマイナス
        score_stats["profit_factor"] < 1.1, # プロフィットファクター（総純利益 / 総損失）が1.1以下
    ]

    if any(ignore):
        score = 0
    else:
        score = sum(score_stats["trade"]) * (sum(score_stats["gain"]) / 10000) * (1 - max(score_stats["max_drawdown"]))

    print_score_stats("short", score, score_stats, simulator_setting.assets, strategy_setting)

    return score

def simulate_by_term(param):
    strategy_simulator, strategy_setting, data, start_date, end_date = param
    simulators = strategy_simulator.simulates(strategy_setting, data, start_date, end_date)
    return strategy_simulator.get_stats(simulators, start_date, end_date)

def select_codes(args, start, end, strategy_simulator):
    codes = strategy_simulator.select_codes(args, start, end)
    return codes

def target_codes(args, terms, strategy_simulator):
    codes = []
    for term in terms:
        start = utils.to_format(term["start_date"])
        end = utils.to_format(term["end_date"])
        targets = select_codes(args, start, end, strategy_simulator)
        codes = list(set(codes + targets))
    return codes

def select_data(codes, stocks, start, end):
    select = {"data": {}, "index": stocks["index"], "args": stocks["args"]}

    args = select["args"]

    for code in codes:
        if not code in stocks["data"].keys():
            continue
        start_date = utils.to_format(utils.to_datetime(start) - utils.relativeterm(3))
        select["data"][code] = stocks["data"][code].split(start_date, end)
#        print(select["data"][code].daily["date"].astype(str).values)

    return select

def simulate_params(stocks, terms, strategy_simulator):
    params = []
    strategy_simulator.simulator_setting.strategy = None
    for term in terms:
        start = utils.to_format(term["start_date"])
        end = utils.to_format(term["end_date"])
        codes = select_codes(stocks["args"], start, end, strategy_simulator)
        select = select_data(codes, stocks, start, end)
        params.append((select, utils.to_format(utils.to_datetime(start)), end))
    return params

# 1つの設定でstart~endまでのterm毎のシミュレーション結果を返す
def simulate_by_multiple_term(stocks, params):
    try:
        p = Pool(int(stocks["args"].jobs))
        performances = p.map(simulate_by_term, params)
    except KeyboardInterrupt:
        p.close()
        exit()
    finally:
        p.close()
    return performances

# パラメータ評価用の関数
# 指定戦略を全銘柄に対して最適化
def objective(args, strategy_setting, stocks, params, validate_params, strategy_simulator):
    print(strategy_setting.__dict__, strategy_simulator.combination_setting.seed)
    try:
        params = list(map(lambda x: (strategy_simulator, strategy_setting) + x, params))
        performances = simulate_by_multiple_term(stocks, params)
        score = get_score(args, performances, strategy_simulator.simulator_setting, strategy_setting)
        validate_score = 0
        if score > 0:
            validate_params = list(map(lambda x: (strategy_simulator, strategy_setting) + x, validate_params))
            validate_performances = simulate_by_multiple_term(stocks, validate_params)
            validate_score = get_score(args, validate_performances, strategy_simulator.simulator_setting, strategy_setting)

        if validate_score > 0:
            score = score * validate_score
        else:
            score = score + validate_score
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
    n_random_starts = int(args.n_calls/4) if args.n_calls > 100 else 10
    random_state = int(time.time()) if args.random > 0 else None
    res_gp = gp_minimize(
        lambda x: objective(args, strategy_setting.by_array(x), stocks, params, validate_params, strategy_simulator),
        space, n_calls=args.n_calls, n_initial_points=n_random_starts, random_state=random_state)
    result = strategy_setting.by_array(res_gp.x)
    score = res_gp.fun
    print("done strategy_optimize: %s random_state:%s" % (utils.timestamp(), random_state))
    return result, score

def create_terms(args):
    optimize_terms = []
    validate_terms = []

    valid_end_date = utils.to_datetime(args.date)
    for c in range(args.count):
        if args.instant:
            end_date = valid_end_date - utils.relativeterm(args.validate_term, with_time=True)
            start_date = end_date - utils.relativeterm(args.validate_term*args.optimize_count, with_time=True)
        else:
            end_date = valid_end_date - utils.relativeterm(args.validate_term)
            start_date = end_date - utils.relativeterm(args.validate_term*args.optimize_count)

        term = {"start_date": start_date, "end_date": end_date - utils.relativeterm(1, with_time=True)}
        validate_term = {"start_date": end_date, "end_date": valid_end_date - utils.relativeterm(1, with_time=True)}

        if args.optimize_count > 0:
            optimize_terms.append(term)
        validate_terms.append(validate_term)

        valid_end_date = start_date
    print(list(map(lambda x: "%s - %s" % (str(x["start_date"]), str(x["end_date"])), optimize_terms)))
    print(list(map(lambda x: "%s - %s" % (str(x["start_date"]), str(x["end_date"])), validate_terms)))

    optimize_terms = sorted(optimize_terms, key=lambda x: x["start_date"])
    validate_terms = sorted(validate_terms, key=lambda x: x["start_date"])

    return optimize_terms, validate_terms

def output_performance(args, performances):
    filename = "%s/performances/%sperformance.json" % (args.output_dir, strategy.get_prefix(args))
    with open(filename, "w") as f:
        f.write(json.dumps(to_jsonizable(performances)))

def create_performance(args, simulator_setting, performances):
    if len(performances) == 0:
        return {}

    # 簡易レポート
    for date, performance in sorted(performances.items(), key=lambda x: utils.to_datetime(x[0])):
        pickup = ["gain", "max_unrealized_gain", "crash", "drawdown", "max_drawdown", "auto_stop_loss"]
        stats = list(map(lambda x: "%s: %.02f" % (x, performance[x]), pickup))
        print(date, ",\t".join(stats))

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
        "commission": sum(list(map(lambda x: x["commission"], performances.values()))),
        "oneday_commission": sum(list(map(lambda x: x["oneday_commission"], performances.values()))),
        "interest": sum(list(map(lambda x: x["interest"], performances.values()))),
        "auto_stop_loss": sum(list(map(lambda x: x["auto_stop_loss"], performances.values()))),
        "trade": sum(list(map(lambda x: x["trade"], performances.values()))),
        "win_trade": sum(list(map(lambda x: x["win_trade"], performances.values()))),
    }
    print(json.dumps(to_jsonizable(result)))

    return result

def output_setting(args, strategy_settings, strategy_simulator, score, optimize_score, validate_score, optimize_report, validate_report, performance_score):
    filename = "%s/%s" % (args.output_dir, strategy.get_filename(args))
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, "w") as f:
        f.write(json.dumps(to_jsonizable({
            "date": args.date,
            "term": args.validate_term,
            "score": int(score) * float(performance_score),
            "optimize_score": int(optimize_score),
            "validate_score": int(validate_score),
            "max_position_size": strategy_simulator.combination_setting.max_position_size,
            "position_sizing": strategy_simulator.combination_setting.position_sizing,
            "stop_loss_rate": strategy_simulator.simulator_setting.stop_loss_rate,
            "taking_rate": strategy_simulator.simulator_setting.taking_rate,
            "min_unit": strategy_simulator.simulator_setting.min_unit,
            "setting": list(map(lambda x: x.__dict__, strategy_settings)),
            "seed": strategy_simulator.combination_setting.seed,
            "use_limit": args.use_limit,
            "auto_stop_loss": args.auto_stop_loss,
            "condition_size": strategy_simulator.combination_setting.condition_size,
            "passive_leverage": args.passive_leverage,
            "ensemble_dir": args.ensemble_dir,
            "weights": strategy_simulator.combination_setting.weights,
            "optimize_report": optimize_report,
            "validate_report": validate_report,
            "performance_score": float(performance_score)
        })))

def get_performance_score(optimize_performances, validate_performances):
    performances = {}
    performances.update(optimize_performances)
    performances.update(validate_performances)

    sum_gain = sum(list(map(lambda x: x["gain"], performances.values()))) # 総利益
    sum_trade = sum(list(map(lambda x: x["trade"], performances.values()))) # 総トレード数
    ave_trade = numpy.average(list(map(lambda x: x["trade"], performances.values()))) # 平均トレード数
    gain_per_trade = 0 if sum_trade == 0 else (sum_gain / sum_trade)# 1トレード当たりの利益

    average_line = [i*ave_trade * gain_per_trade for i in range(1, len(performances)+1)]

    gain = 0
    diff = []
    for ave, d in zip(average_line, sorted(performances.items(), key=lambda x:utils.to_datetime(x[0]))):
        gain = gain + d[1]["gain"]
        diff = diff + [abs(ave - gain)]

    score = 0 if sum(diff) == 0 else 1 / sum(diff)

    return score

def validation(args, stocks, terms, strategy_simulator, combination_setting, strategy_settings):
    performances = {}
    strategy_simulator.strategy_settings = strategy_settings
    strategy_simulator.combination_setting = combination_setting
    params = simulate_params(stocks, terms, strategy_simulator)
    simulator_setting = copy.deepcopy(strategy_simulator.simulator_setting)

    if args.verbose or args.apply_compound_interest:
        print("debug mode")
        for param in params:
            _, start_date, end_date = param
            result = simulate_by_term((strategy_simulator, strategy_settings[-1]) + param)

            if args.apply_compound_interest: # 複利を適用
                strategy_simulator.simulator_setting.assets += result["gain"]
                print("assets:", strategy_simulator.simulator_setting.assets, result["gain"])

            performances[utils.to_format(utils.to_datetime_by_term(end_date))] = result
    else:
        params = list(map(lambda x: (strategy_simulator, strategy_settings[-1]) + x, params))
        results = simulate_by_multiple_term(stocks, params)
        for result in results:
            performances[utils.to_format(utils.to_datetime_by_term(result["end_date"]))] = result

    # 検証スコア
    score = -get_score(args, performances.values(), strategy_simulator.simulator_setting, strategy_settings[-1])

    # 結果の表示 =============================================================================
    report = create_performance(args, simulator_setting, performances)

    return score, report, performances

def walkforward(args, stocks, terms, validate_terms, strategy_simulator, combination_setting):
    default_weights = copy.deepcopy(combination_setting.weights)
    if args.verbose:
        print("weights", default_weights)

    is_optimize = args.optimize_count > 0 and not args.ignore_optimize

    # 最適化
    if is_optimize:
        strategy_simulator.combination_setting = copy.deepcopy(combination_setting)

        if args.use_optimized_init == 0:
            strategy_simulator.combination_setting.seed = [int(time.time())]
        else:
            strategy_simulator.combination_setting.seed = combination_setting.seed[:args.use_optimized_init] + [int(time.time())]

        conditions_index = strategy_simulator.strategy_creator(args).conditions_index()
        params = simulate_params(stocks, terms, strategy_simulator)
        validate_params = simulate_params(stocks, validate_terms, strategy_simulator)
        strategy_setting, score = strategy_optimize(args, stocks, params, validate_params, strategy_simulator)

        objective(args, strategy_setting, stocks, params, validate_params, strategy_simulator) # 選ばれた戦略スコアを表示するため
        strategy_settings = strategy_simulator.strategy_settings[:args.use_optimized_init] + [strategy_setting]
        validate_combination_setting = copy.deepcopy(strategy_simulator.combination_setting)
        print(strategy_setting.__dict__, score)
        weights = update_weights(conditions_index, default_weights, score, amount=args.amount)
    else:
        _, strategy_settings = strategy.load_strategy_setting(args)
        if args.output:
            print("Need -o parameter or Using --ignore_optimize. don't output simulate setting.")
            exit()
        validate_combination_setting = copy.deepcopy(combination_setting)
        weights = default_weights

    # 検証
    validate_combination_setting.use_limit = False
    print("===== [optimize] =====")
    optimize_score, optimize_report, optimize_performances = validation(args, stocks, terms, strategy_simulator, validate_combination_setting, strategy_settings)
    print("===== [validate] =====")
    validate_score, validate_report, validate_performances = validation(args, stocks, validate_terms, strategy_simulator, validate_combination_setting, strategy_settings)

    # レポート出力
    if args.performance:
        performances = {}
        performances.update(optimize_performances)
        performances.update(validate_performances)
        output_performance(args, performances)

    performance_score = get_performance_score(optimize_performances, validate_performances)
    print("performance_score:", performance_score)

    if args.output:
        print("strategy_setting:", len(strategy_settings))
        output_setting(args, strategy_settings, strategy_simulator, score, optimize_score, validate_score, optimize_report, validate_report, performance_score)

    return weights 

def update_weights(conditions_index, weights, score, amount):
    for method, indexies in conditions_index.items():
        if not method in weights.keys():
            weights[method] = {}
        for index in indexies:
            i = str(index)
            if i in weights[method].keys():
                if score < 0:
                    weights[method][i] = weights[method][i] + amount
                else:
                    if weights[method][i] > 0:
                        weights[method][i] = math.ceil(weights[method][i] / 2)
                    else:
                        weights[method][i] = math.ceil(weights[method][i] * 2)
            else:
                if score < 0:
                    weights[method][i] = amount
                else:
                    weights[method][i] = -math.ceil(amount / 10)
    return weights

######################################################################
print(utils.timestamp())
proc_start_time = time.time()

simulate_setting = create_setting(args)

if args.optimize_count > 0 and not args.ignore_optimize and args.use_optimized_init == 0 and not args.with_weights:
    combination_setting = strategy.create_combination_setting(args, use_json=False)
    strategy_settings = []
else:
    combination_setting = strategy.create_combination_setting(args)
    _, strategy_settings = strategy.load_strategy_setting(args)

combination_setting.montecarlo = args.montecarlo

strategy_simulator = StrategySimulator(simulate_setting, combination_setting, strategy_settings, verbose=args.verbose)

# 翌期間用の設定を出力する
# 都度読み込むと遅いので全部読み込んでおく
optimize_terms, validate_terms = create_terms(args)
min_start_date = min(list(map(lambda x: x["start_date"], optimize_terms + validate_terms)))
start = utils.to_format_by_term(min_start_date)
end = utils.to_datetime(args.date)
end = utils.to_format_by_term(end)
codes = target_codes(args, optimize_terms + validate_terms, strategy_simulator)

print("target : %s" % codes, start, end)

stocks = load(args, list(set(codes)), optimize_terms + validate_terms, combination_setting)

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

    if not args.ignore_default:
        filename = strategy.get_filename(args)
        params = ["cp", "%s/%s" % (default_output_dir, filename), "%s/tmp/default_%s" % (args.output_dir, filename)]
        status = subprocess.call(params)

    if status == 0 and args.skip_optimized:
        print("skip. optimized.")
        exit()

    if args.with_weights:
        strategy_simulator.combination_setting.weights = update_weights(strategy_simulator.strategy_creator(args).conditions_index(), combination_setting.weights, score=-1, amount=args.amount)

    for i in range(args.random):
        combination_setting.weights = walkforward(args, stocks, optimize_terms, validate_terms, strategy_simulator, combination_setting)

        params = ["cp", "%s/%s" % (args.output_dir, filename), "%s/tmp/%s_%s" % (args.output_dir,i, filename)]
        subprocess.call(params)

    params = ["sh", "simulator/copy_highest_score_setting.sh", strategy.get_prefix(args), args.output_dir]
    subprocess.call(params)
else:
    walkforward(args, stocks, optimize_terms, validate_terms, strategy_simulator, combination_setting)

print(utils.timestamp())
proc_end_time = time.time()
print("proc time: %s" % (proc_end_time - proc_start_time))
