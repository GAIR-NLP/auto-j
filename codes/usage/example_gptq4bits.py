from transformers import AutoModelForCausalLM, AutoTokenizer
from constants_prompt import *

path = "GAIR/autoj-13b-GPTQ-4bits" # or your local path to auto-j-4bits

tokenizer = AutoTokenizer.from_pretrained(path)

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")

query = "<your query>"
response = "<a response>"
text = build_autoj_input(query, response)

# or for pairwise, you can ->
# response_another = "<another response>"
# text = build_autoj_input(query, response, response_another, "pairwise_tie")

inputs = tokenizer(text, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_length=1000, temperature=0.0, do_sample=False, top_p=1.0)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# note that this output contains the input part, you may need to remove it by yourself
