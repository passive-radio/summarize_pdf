import os
import langchain
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate

import secret

langchain.verbose = True
os.environ["OPENAI_API_KEY"] = secret.OPENAI_API_KEY

llm = OpenAI(model_name="text-davinci-003", temperature=0)

template = """
あなたは加算・減算を絶対間違えない計算機です。
以下の問題に答えます。ただし以下の制約を守って出力して下さい。

制約```
1. ステップバイステップで思考すること。
2. ステップバイステップで思考した内容を出力すること。
3. 出力の最後は問題の答えであること。
4. 計算を間違えないこと。
```

### 問題 ###
{command}
### 問題終了 ###
"""

cot_prompt = PromptTemplate(
    input_variables=["command"],
    template = template,
)

cot_chain = LLMChain(llm=llm, prompt=cot_prompt)

summarize_template = """
入力を一言の結論に要約してください。

### 入力 ###
{input}
### 入力終了 ###
"""

summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template=summarize_template,
)
summarize_chain = LLMChain(
    llm=llm, prompt=summarize_prompt
)
cot_summarize_chain = SimpleSequentialChain(
    chains=[cot_chain, summarize_chain]
)

question = "健志くんとたかし君が駄菓子屋でうまい棒を10本ずつ購入した。その後、健志くんはたかし君に2本あげました。だけど、食べきれないと思ったたかし君は、健志くんに3本あげました。この後に、健志くんが持っていたうまい棒の本数は何本ですか？"

ret = cot_summarize_chain(
    question
)

print(ret["output"])