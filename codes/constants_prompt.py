PROMPT_INPUT_SYSTEM: str = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'

PROMPT_INPUT_WO_SYSTEM: str = "[INST] {input} [/INST]"

PROMPT_INPUT_FOR_SCENARIO_CLS: str = "Identify the scenario for the user's query, output 'default' if you are uncertain.\nQuery:\n{input}\nScenario:\n"

single = """Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {prompt}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

pairwise_tie = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""

protocol_mapping = {
    "pairwise_tie": pairwise_tie,
    "single": single,
}


def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)


def build_autoj_input(prompt, resp1, resp2=None, protocol="single"):
    user_msg = protocol_mapping[protocol].format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg, )


if __name__ == '__main__':
    t = build_autoj_input("instruction", "resp1", "resp2", "pairwise_tie")
    print(t)
