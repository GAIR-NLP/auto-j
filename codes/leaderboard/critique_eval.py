import argparse
from utils_constants import *


def resolve_decision(decision):
    sent1 = decision.split("\n")[0]
    # check at most one of the following are in the first sentence "A:", "B:", or "C:"
    found = False
    if "A:" in sent1:
        decision = 0
        found = True
    if "B:" in sent1:
        if found: raise ValueError
        decision = 1
        found = True
    if "C:" in sent1:
        if found: raise ValueError
        decision = 2
        found = True
    if not found:
        raise ValueError("Cannot resolve decision")
    return decision


def extract_vote_from_decision(vote, exchange_tag):
    if exchange_tag: vote = 1 - vote if vote != 2 else 2
    return vote

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="../../data/test/testdata_critique.jsonl")
    parser.add_argument("--openai_comparison_file", type=str, default="../../data/outputs/gpt-4-Eval_auto-j_vs_chatgpt.jsonl")
    args = parser.parse_args()

    source = read_jsonl(args.source_file)
    data = read_jsonl(args.openai_comparison_file)

    ress = [0, 0, 0]

    scenarios = {}

    limit = 0

    for x, s, in zip(data, source):
        scenario = s['scenario']
        output = x['output']

        exchange_tag = x['meta']['exchange'] if "meta" in x else False

        vote = extract_vote_from_decision(resolve_decision(output.split("\n")[0]), exchange_tag)

        ress[vote] += 1

        if scenario not in scenarios:
            scenarios[scenario] = {0: 0, 1: 0, 2: 0}
        scenarios[scenario][vote] += 1

    results = {}


    for gname, glist in scenario_group.items():
        ress_group = [0, 0, 0]
        for x in glist:
            ress_group = [ress_group[i] + scenarios[x][i] for i in range(3)]
        winrate = round(100*ress_group[0] / sum(ress_group),2)
        results[gname] = winrate


    # overall winrate
    winrate = round(100*ress[0] / sum(ress),2)
    results["Overall"] = winrate

    print("Group\tWinrate")
    print("---------------")
    for k,v in results.items():
        if k=="Overall":
            print("---------------")
        print(f"{k}\t{v}")