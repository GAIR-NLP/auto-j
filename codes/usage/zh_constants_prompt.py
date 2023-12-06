PROMPT_INPUT_SYSTEM: str = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'

PROMPT_INPUT_WO_SYSTEM: str = "[INST] {input} [/INST]"

PROMPT_INPUT_FOR_SCENARIO_CLS: str = "Identify the scenario for the user's query, output 'default' if you are uncertain.\nQuery:\n{input}\nScenario:\n"


zh_single = """为上传的针对给定用户问题的回应撰写评论, 并为该回复打分:

[BEGIN DATA]
***
[用户问询]: {prompt}
***
[回应]: {response}
***
[END DATA]

为这个回应撰写评论. 在这之后, 你应该按照如下格式给这个回应一个最终的1-10范围的评分: "[[评分]]", 例如: "评分: [[5]]"."""


zh_pairwise_tie = """你将要评价两个针对给定用户问题的回应，并判断哪个回应胜出或两者持平。以下是相关内容:

[BEGIN DATA]
***
[用户问询]: {prompt}
***
[回应1]: {response}
***
[回应2]: {response_another}
***
[END DATA]

以下是评价并比较两个回应的指南：

1. 确定区分两个回应的关键因素。
2. 通过提供最终对何者胜出或两者持平的决定来总结你的比较和评价。最终决定语句应以\"所以，最终决定是回应1/回应2/平局\"开头。确保你的决定与你提供的综合的评价和比较一致。"""

zh_protocol_mapping = {
    "zh_pairwise_tie": zh_pairwise_tie,
    "zh_single": zh_single,
}


def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)


def zh_build_autoj_input(prompt, resp1, resp2=None, protocol="single"):
    user_msg = zh_protocol_mapping[protocol].format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg, )


if __name__ == '__main__':
    t = zh_build_autoj_input("instruction", "resp1", "resp2", "zh_pairwise_tie")
    print(t)
