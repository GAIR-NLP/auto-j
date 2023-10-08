from vllm import LLM, SamplingParams
import torch
from constants_prompt import build_autoj_input

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    model_name_or_dir = "GAIR/autoj-13b"  # or "local path to auto-j"
    llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    input_pairwise = build_autoj_input(prompt="what is 1+0?",
                                       resp1="1+0 is 11",
                                       resp2="the answer is 1",
                                       protocol="pairwise_tie")  # for pairwise response comparison
    input_single = build_autoj_input(prompt="what is 1+0?",
                                     resp1="1",
                                     resp2=None, protocol="single")  # for single response evaluation

    input = input_pairwise  # or input_single

    outputs = llm.generate(input, sampling_params)

    print(outputs[0].outputs[0].text)
