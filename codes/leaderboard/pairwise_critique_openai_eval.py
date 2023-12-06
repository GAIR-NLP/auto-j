import argparse
from utils_constants import *
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="../../data/test/testdata_critique.jsonl")
    parser.add_argument("--openai_model", type=str, default="gpt-4")  # or gpt-3.5-turbo
    parser.add_argument("--critic_file", type=str, default="../../data/outputs/critique_example_output.jsonl")
    parser.add_argument("--critic_name", type=str, default="auto-j")
    parser.add_argument("--reference_file", type=str, default="../../data/test/reference_chatgpt_critique.jsonl")
    parser.add_argument("--openai_api", type=str, default=None)
    parser.add_argument("--openai_org", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--fix_mode", action='store_true')
    args = parser.parse_args()

    # set OpenAI things
    if args.openai_api is None:
        args.openai_api = os.environ.get("OPENAI_API_KEY")
    if args.openai_org is not None:
        openai.organization = args.openai_org

    # prepare data, output file and other things
    source = read_jsonl(args.source_file)
    critiques_challenger = read_jsonl(args.critic_file)
    critiques_reference = read_jsonl(args.reference_file)
    assert len(source) == len(critiques_challenger) == len(critiques_reference)

    output_file = f"../../data/outputs/{args.openai_model}-Eval_{args.critic_name}_vs_chatgpt.jsonl"

    already_have = [] if not os.path.exists(output_file) else read_jsonl(output_file)
    start_from = len(already_have)
    already_have += [None] * (len(source) - len(already_have))

    assert len(source) == len(critiques_challenger) == len(critiques_reference) == len(already_have)

    all_inputs = []
    all_metas = []

    assert args.language in ["English", "Chinese"]
    # prepare inputs for GPT-4 / ChatGPT
    for idx, (a, b, c, d) in enumerate(zip(source, critiques_challenger, critiques_reference, already_have)):
        prompt, response, scenario = a["prompt"], a["response"], a["scenario"]

        feedback1, feedback2 = b["output"], c["output"]

        exchange_tag = not d['meta']['exchange'] if d is not None else random.choice([True, False])
        # for fix mode, i.e. fix the "Failed!" cases in an existing file, we use the exchange tag in the file
        if args.fix_mode:
            exchange_tag = d['meta']['exchange']
            if d['output'] != "Failed!": continue  # skip the ones that are not failed

        if exchange_tag: feedback1, feedback2 = feedback2, feedback1

        all_metas.append({"exchange": exchange_tag, "idx": idx})

        if args.language == "English":
            input = critique_eval_prompt.format(prompt=prompt, response=response, feedback1=feedback1,feedback2=feedback2)
        else:
            input = zh_critique_eval_prompt.format(prompt=prompt, response=response, feedback1=feedback1, feedback2=feedback2)

        all_inputs.append({"usermsg": input})

    engine = OpenAIChat(
        api_key=args.openai_api,
        model=args.openai_model, temperature=0.0, max_tokens=2048, top_p=1.0,
        presence_penalty=0.0, frequency_penalty=0.0, request_timeout=120)

    if args.fix_mode:
        print_colored_text("Fix mode is on, we will fix the \"Failed!\" cases in the existing file.", "yellow")
        # we fix the "Failed!" ones
        outputs = engine.generate_batch(all_inputs)
        # write back to the file
        for idx, output in enumerate(outputs):
            print_colored_text(all_metas[idx]["idx"], 'green')
            elegant_show(output)
            print_colored_text('-----------------------------', "yellow")
            already_have[all_metas[idx]["idx"]]['output'] = output['output']
            already_have[all_metas[idx]["idx"]]['meta']['exchange'] = all_metas[idx]['exchange']
            already_have[all_metas[idx]["idx"]]['cost'] = output['cost']
            already_have[all_metas[idx]["idx"]]['finish_reason'] = output['finish_reason']
        write_jsonl(already_have, output_file)
    else:
        print_colored_text("Fix mode is off, we will generate all the cases, start from {}.".format(start_from), "yellow")
        batched_generate_with_write(engine, all_inputs,
                                    output_file_name=output_file,
                                    batch_size=args.batch_size,
                                    already_have=start_from, final_metas=all_metas, )
