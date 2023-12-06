import json

# with open ("./pairwise_merged.jsonl", "r") as file:
with open("./zh_reference_chatgpt_critique.jsonl") as file:
    data = file.readlines()

for single_data in data:
    single_data = json.loads(single_data)
    # with open ("./bilin_paiwise_translated_show.txt", "a") as f:
    # with open ("/cpfs01/user/liupengfei/yguo/zn_autoj/autoj/data/outputs/gpt4_comparison_show.txt", "a") as f:
    with open("./_show1.txt", "a") as f:
        print (single_data, file=f)

