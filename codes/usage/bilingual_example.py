from vllm import LLM, SamplingParams
import torch
from constants_prompt import build_autoj_input
from zh_constants_prompt import zh_build_autoj_input
import argparse


def extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            pred_label = 0
        elif pred_rest.startswith('response 2'):
            pred_label = 1
        elif pred_rest.startswith('tie'):
            pred_label = 2
    return pred_label


def zh_extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('最终决定是')
    pred_label = -1
    if pos != -1:
        # pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        pred_rest = raw_output[pos + len('最终决定是'):].strip()
        if pred_rest.startswith('回应1'):
            pred_label = 0
        elif pred_rest.startswith('回应2'):
            pred_label = 1
        elif pred_rest.startswith('平局'):
            pred_label = 2
    return pred_label


def extract_single_rating(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0


def zh_extract_single_rating(score_output):
    if "评分:[[" in score_output:
        pos = score_output.rfind("评分:[[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("评分:[["):pos2].strip())
    elif "打分:[[" in score_output:
        pos = score_output.rfind("打分:[[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("打分:[["):pos2].strip())
    else:
        return 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="Chinese", help="Choose Chinese or English evaluation")
    args = parser.parse_args()
    assert args.language == "Chinese" or args.language == "English"
    num_gpus = torch.cuda.device_count()
    model_name_or_dir = "GAIR/autoj-bilingual-6b"  # or "local path to auto-j"
    llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus,)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)


    if args.language == "Chinese":
        # input_single = zh_build_autoj_input(prompt="中国的首都是哪里？",
        #                                 resp1="中国的首都是上海。",
        #                                 protocol="zh_single")
        input_pairwise = zh_build_autoj_input(prompt="中国的首都是哪里?",
                                        resp1="中国的首都是北京。",
                                        resp2="中国的首都是上海。",
                                        protocol="zh_pairwise_tie")
        input = input_pairwise  # or input_single
        # input = input_single

        outputs = llm.generate(input, sampling_params)

        judgment = outputs[0].outputs[0].text

        print(judgment)

        evaluation_result = zh_extract_pariwise_result(judgment)
        # single_score = zh_extract_single_rating(judgment)
        # print(single_score)

        print(evaluation_result)

    elif args.language == "English":
        input_pairwise = build_autoj_input(prompt="What's the capital of the United States?",
                                       resp1="The capital of the United States is New York.",
                                       resp2="The captical of the United States is Washington D.C.",
                                       protocol="pairwise_tie")

            # input = input_pairwise  # or input_single
        input = input_pairwise

        outputs = llm.generate(input, sampling_params)

        judgment = outputs[0].outputs[0].text

        print(judgment)

        evaluation_result = extract_pariwise_result(judgment)

        print(evaluation_result)
