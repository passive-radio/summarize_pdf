import os
import langchain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, ConversationChain
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

import secret

langchain.verbose = False
os.environ["OPENAI_API_KEY"] = secret.OPENAI_API_KEY

llm = OpenAI(model_name="text-davinci-003", temperature=0)

conversation = ConversationChain(
    llm = llm, memory= ConversationBufferMemory()
)

while True:
    with get_openai_callback() as cb:
        user_messasge = input("You: ")
        ret = conversation.predict(input=user_messasge)
        print(f"AI: {ret}")
        print(cb)
        
        costs = [f"Total Tokens: {cb.total_tokens}\n",
                 f"Prompt Tokens: {cb.prompt_tokens}\n", 
                 f"Completion Tokens: {cb.completion_tokens}\n", 
                 f"Total Cost (USD): ${cb.total_cost}\n"]
        
        with open("out.md", "w", ) as f:
            f.write("# Prompt Cost\n")
            f.writelines(costs)
            f.write("\n")
        
        with open("out.md", "a") as f:
            f.write(ret)