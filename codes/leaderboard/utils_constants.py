import json
import os
import openai
import ast
import asyncio
from typing import Union
import tiktoken
from tqdm import tqdm

scenario_group = {
    "Summarization": ["post_summarization", "text_summarization", "note_summarization"],
    "Exam Questions": ["math_reasoning", "solving_exam_question_with_math", "solving_exam_question_without_math", ],
    "Code": ["code_simplification",
             "code_generation",
             "explaining_code",
             "code_correction_rewriting",
             "code_to_code_translation",
             ],
    "Rewriting": [
        "text_simplification",
        "language_polishing",
        "instructional_rewriting",
        "text_correction",
        "paraphrasing",
    ],
    "Creative Writing": ["writing_song_lyrics",
                         "writing_social_media_post", "writing_blog_post", "writing_personal_essay",
                         "creative_writing", "writing_advertisement", "writing_marketing_materials",
                         "writing_presentation_script",
                         "counterfactual", ],
    "Functional Writing": [
        "writing_product_description",
        "writing_job_application",
        "writing_news_article",
        "writing_biography",
        "writing_email",
        "writing_legal_document",
        "writing_technical_document",
        "writing_scientific_paper",
        "functional_writing",
        "writing_cooking_recipe",
    ],
    "General Communication": ["asking_how_to_question", "open_question", "analyzing_general", "explaining_general",
                              "seeking_advice", "recommendation", "value_judgement", "verifying_fact", "chitchat",
                              "roleplay",
                              "planning", "brainstorming",
                              ],
    "NLP Tasks": [
        "ranking",
        "text_to_text_translation",
        "data_analysis",
        "classification_identification",
        "title_generation",
        "question_generation",
        "reading_comprehension",
        "keywords_extraction",
        "information_extraction",
        "topic_modeling",
        "others",
    ],
}


critique_eval_prompt = """You are a helpful and precise assistant for checking the quality of the feedback.
Two pieces of feedback have been provided for the same response to a particular query. Which one is better with regard to their correctness, comprehensiveness, and specificity to the query?

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response]: {response}
***
[Feedback 1]: {feedback1}
***
[Feedback 2]: {feedback2}
***
[END DATA]

Please choose from the following options, and give out your reason in the next line. 
A: Feedback 1 is significantly better.
B: Feedback 2 is significantly better.
C: Neither is significantly better."""

zh_critique_eval_prompt = """你是一个乐于助人且回答准确的助手，将要来评估反馈的质量。
我向你提供了两条针对特定用户问询的相同回答的反馈。就它们的正确性、全面性和与问询的相关性而言，哪一条更好？

[BEGIN DATA]
***
[用户问询]: {prompt}
***
[回应]: {response}
***
[反馈1]: {feedback1}
***
[反馈2]: {feedback2}
***
[END DATA]

请在以下选项中做出选择，并在这之后的一行给出你的理由
A:反馈1明显更好
B:反馈2明显更好
C:并没有哪条反馈明显更好"""


reversed_scenario_group = {
    vv: k for k, v in scenario_group.items() for vv in v
}


def elegant_show(something, level=0, sid=0, full=False):
    # str,float,int
    # all print in this call should add level*4 spaces
    prefix = "\t" * level

    if isinstance(something, (str, float, int)) or something is None:
        if isinstance(something, str):
            # if '\n' in something:
            #     something = '\n'+something
            # add prefix whenever go to a new line in this string
            something = something.replace("\n", f"\n{prefix}")
        print(prefix, f"\033[1;35mElement: \033[0m", something)
    elif isinstance(something, list) or isinstance(something, tuple):
        # take a random example, and length
        # sid = 0
        if len(something) == 0:
            print(prefix,
                  f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& No elements! \033[0m")
        elif not full or len(something) == 1:
            print(prefix,
                  f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& first element ...\033[0m")
            elegant_show(something[sid], level + 1, sid, full)
        else:
            print(prefix,
                  f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& Elements ...\033[0m")
            for i in range(len(something) - 1):
                elegant_show(something[i], level + 1, sid, full)
                print(prefix + '\t', f"\033[1;33m-------------------------------\033[0m")
            elegant_show(something[-1], level + 1, sid, full)

    elif isinstance(something, dict):
        for k, v in something.items():
            print(prefix, f"\033[1;34mKey: \033[0m{k} \033[1;34m...\033[0m")
            elegant_show(v, level + 1, sid, full)
    else:
        print(prefix, f"\033[1;31mError @ Type: \033[0m{type(something)}")
        raise NotImplementedError


def read_jsonl(jsonl_file_path):
    s = []
    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        linex = line.strip()
        if linex == '':
            continue
        s.append(json.loads(linex))
    return s


def write_jsonl(data, jsonl_file_path, mode="w"):
    # data is a list, each of the item is json-serilizable
    assert isinstance(data, list)
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))
    with open(jsonl_file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def num_tokens_from_string(inputs: Union[str, list], encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string or a list of string."""
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list):
        raise ValueError(f"string must be a string or a list of strings, got {type(inputs)}")
    num_tokens = 0
    encoding = tiktoken.get_encoding(encoding_name)
    for xstring in inputs:
        num_tokens += len(encoding.encode(xstring))
    return num_tokens


def calculate_openai_api_cost(usage, model_name):
    """
    This function is used to calculate the cost of a request.
    :param usage:
    :param model_name:
    :return:
    """
    mapping = {
        "gpt-3.5-turbo": (0.0015, 0.002),
        "gpt-3.5-turbo-16k": (0.003, 0.004),
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
    }
    intokens = usage.prompt_tokens
    outtokens = usage.completion_tokens

    assert model_name in mapping.keys()
    return mapping[model_name][0] * intokens / 1000 + mapping[model_name][1] * outtokens / 1000


def print_colored_text(text, color="yellow", end=None):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }

    color_code = colors.get(color.lower(), colors["reset"])
    print(f"{color_code}{text}{colors['reset']}", end=end)


def safe_write_jsonl(data, jsonl_file_path, mode="w"):
    # check if the directory exists, if not we create it
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))
    write_jsonl(data, jsonl_file_path, mode)


def batched_generate_with_write(engine, final_inputs, output_file_name, batch_size=-1, already_have=0, final_metas=None):
    total_cost = 0.0
    if final_metas is not None:
        assert len(final_inputs) == len(final_metas)
    if batch_size != -1:
        print_colored_text(f"[INFO] Batched generation with batch size {batch_size}.", "green")
        for batch_start in range(already_have, len(final_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(final_inputs))
            batch = final_inputs[batch_start:batch_end]
            outputs = engine.generate_batch(batch)
            # meta is a list of dict, we put the keys into the output
            if final_metas is not None:
                batch_meta = final_metas[batch_start:batch_end]
                for i in range(len(outputs)):
                    outputs[i]['meta'] = batch_meta[i]
            safe_write_jsonl(outputs, output_file_name, mode='a')
            if 'cost' in outputs[0]:
                total_cost += sum([x['cost'] for x in outputs])
            print_colored_text(
                f"[INFO] Batch {batch_start}-{batch_end}/{len(final_inputs)} are finished and written. | Accumulated total cost: {total_cost}", "green")
    else:
        print_colored_text(f"[INFO] Full generation {len(final_inputs)} samples at one throughput.", "green")
        outputs = engine.generate_batch(final_inputs)
        # meta is a list of dict, we put the keys into the output
        if final_metas is not None:
            for i in range(len(outputs)):
                outputs[i]['meta'] = final_metas[i]
        safe_write_jsonl(outputs, output_file_name, mode='a')
        if "cost" in outputs[0]:
            total_cost = sum([x['cost'] for x in outputs])
        print_colored_text(f"[INFO] All are finished and written.", "green")
        print_colored_text(f"[INFO] Accumulated total cost: {total_cost}", "green")


class OpenAIChat():
    """
    This class is a more complex wrapper for OpenAI API, support async batch generation.
    """

    def __init__(self, api_key=None, model='gpt-3.5-turbo',
                 temperature=0.0, max_tokens=2048, top_p=1.0,
                 frequency_penalty=0, presence_penalty=0, request_timeout=120, ):
        if model == 'gpt-3.5-turbo':
            self.max_length = 4096
        elif model == 'gpt-3.5-turbo-16k':
            self.max_length = 16384
        elif model == 'gpt-4':
            self.max_length = 8192
        elif model == 'gpt-4-32k':
            self.max_length = 32768
        else:
            raise ValueError('not supported model!')
        self.config = {'model_name': model, 'max_tokens': max_tokens,
                       'temperature': temperature, 'top_p': top_p,
                       'request_timeout': request_timeout, "frequency_penalty": frequency_penalty,
                       "presence_penalty": presence_penalty}
        openai.api_key = api_key

    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        except:
            return None

    async def dispatch_openai_requests(
            self,
            messages_list,
            enable_tqdm
    ):
        """Dispatches requests to OpenAI API asynchronously.

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """

        async def _request_with_retry(id, messages, retry=3):
            for _ in range(retry):
                try:
                    actual_max_tokens = min(self.config["max_tokens"], self.max_length - 20 - sum(
                        [num_tokens_from_string(m['content']) for m in messages]))
                    if actual_max_tokens < self.config["max_tokens"]:
                        if actual_max_tokens > 0:
                            print(
                                f'Warning: max_tokens is too large, reduce to {actual_max_tokens} due to model max length limit ({self.max_length})!')
                        else:
                            print(f'Input is longer than model max length ({self.max_length}), aborted!')
                            return id, None
                    response = await openai.ChatCompletion.acreate(
                        model=self.config['model_name'],
                        messages=messages,
                        max_tokens=actual_max_tokens,
                        temperature=self.config['temperature'],
                        top_p=self.config['top_p'],
                        request_timeout=self.config['request_timeout'],
                    )
                    return id, response
                except openai.error.RateLimitError:
                    print('Rate limit error, waiting for 40 second...')
                    await asyncio.sleep(40)
                except openai.error.APIError:
                    print('API error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.Timeout:
                    print('Timeout error, waiting for 1 second...')
                    await asyncio.sleep(1)
                except openai.error.ServiceUnavailableError:
                    print('Service unavailable error, waiting for 3 second...')
                    await asyncio.sleep(3)
            return id, None

        async def _dispatch_with_progress():
            async_responses = [
                _request_with_retry(index, messages)
                for index, messages in enumerate(messages_list)
            ]
            if enable_tqdm:
                pbar = tqdm(total=len(async_responses))
            tasks = asyncio.as_completed(async_responses)

            responses = []

            for task in tasks:
                index, response = await task
                if enable_tqdm:
                    pbar.update(1)
                responses.append((index, response))

            if enable_tqdm:
                pbar.close()

            responses.sort(key=lambda x: x[0])  # sort by index

            return [response for _, response in responses]

        return await _dispatch_with_progress()

    async def async_run(self, messages_list, enable_tqdm):
        retry = 1
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
                enable_tqdm=enable_tqdm
            )

            preds = [
                {
                    "output": prediction['choices'][0]['message']['content'],
                    "cost": calculate_openai_api_cost(prediction['usage'], self.config["model_name"]),
                    "finish_reason": prediction['choices'][0]['finish_reason']
                } if prediction is not None else {"output": "Failed!", "cost": 0.0, "finish_reason": "fail"}
                for prediction in predictions
            ]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]

            retry -= 1

        return responses

    def generate_batch(self, msgs, enable_tqdm=True):
        """
        :param msgs: be like [{"sysmsg":"xx","usermsg":"yy"},...]
        :return:
        """
        msg_list = []
        for msg in msgs:
            sysmsg = msg.get("sysmsg", None)
            usermsg = msg.get("usermsg", None)
            assert usermsg is not None
            if sysmsg is None:
                msg_list.append([{"role": "user", "content": usermsg}])
            else:
                msg_list.append(
                    [{"role": "system", "content": msg["sysmsg"]}, {"role": "user", "content": msg["usermsg"]}])
        predictions = asyncio.run(self.async_run(
            messages_list=msg_list,
            enable_tqdm=enable_tqdm
        ))
        # each prediction is a tuple (response, cost)
        return predictions

    def generate_single(self, msg):
        """
        this is just a wrapper for generate_batch when only one msg is given
        :param msg: be like {"sysmsg":"xx","usermsg":"yy"}
        :return:
        """
        sysmsg = msg.get("sysmsg", None)
        usermsg = msg.get("usermsg", None)
        assert usermsg is not None
        if sysmsg is None:
            msg_list = [[{"role": "user", "content": usermsg}]]
        else:
            msg_list = [[{"role": "system", "content": msg["sysmsg"]}, {"role": "user", "content": msg["usermsg"]}]]
        predictions = asyncio.run(self.async_run(
            messages_list=msg_list,
            enable_tqdm=False
        ))
        return predictions[0]
