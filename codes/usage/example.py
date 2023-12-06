from vllm import LLM, SamplingParams
import torch
from constants_prompt import build_autoj_input


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


def extract_single_rating(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    model_name_or_dir = "GAIR/autoj-13b"  # or "local path to auto-j"
    llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)


    input_pairwise = build_autoj_input(prompt="What's the capital of the United States?",
                                       resp1="The capital of the United States is New York.",
                                       resp2="The captical of the United States is Washington D.C.",
                                       protocol="pairwise_tie")  # for pairwise response comparison
    input_single = build_autoj_input(prompt="What's the capital of the United States?",
                                     resp1="The capital of the United States is New York",
                                     resp2=None, protocol="single")  # for single response evaluation

    input = input_pairwise  # or input_single

    outputs = llm.generate(input, sampling_params)

    judgment = outputs[0].outputs[0].text

    print(judgment)

    evaluation_result = extract_pariwise_result(judgment)

    print(evaluation_result)