import os
import langchain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, ConversationChain
from langchain.vectorstores import Chroma
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
import tiktoken

import secret

langchain.verbose = False
os.environ["OPENAI_API_KEY"] = secret.OPENAI_API_KEY

MAX_TOKENS = 1000
company_name = "dena"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=MAX_TOKENS)

query = """与えられた企業の業績に関する文章を以下のフォーマットに必ず従い日本語で出力して下さい。```
# 概要
    1. 企業名
    2. 発表の時期
    3. 発表の要約
    

# 業績のハイライト
    (最小5個、最大10個)
    1. ハイライト1
    2. 
    3. 
    

# 今後の業績見通し (入居率など)
    (最小4個、最大6個)
    1. 見通し1
    2. 
    3. 

# 分類キーワード
    (最小3個、最大4個)
    1. キーワード1
    2. 
    3. 
```"""

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
loader = PyPDFLoader(f'../asset/ir_{company_name}.pdf')
text_splitter = RecursiveCharacterTextSplitter(separators=['\n', '\n\n'],chunk_size=1000, chunk_overlap = 200, length_function=len)

pages = loader.load_and_split()
texts = text_splitter.split_documents(pages[:2])
print("number of orignal PDF pages:", len(pages))
print("number of chunks:", len(texts))

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(pages, embedding=embeddings)

costs = ""
answer = ""

with get_openai_callback() as cb:
    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever = vectorstore.as_retriever(),
        verbose = True,
        )
    
    qa.return_source_documents = True
    print(qa)
    
    
    # print(qa.combine_documents_chain.llm_chain.prompt)
    
    # qa.save("chain.yaml")
    
    # rel = qa._get_docs(query)
    # print(rel)
    # num_tokens = len(encoding.encode(rel))
    # print(num_tokens)
    ret = qa({"query": query})
    print(ret)
    answer = ret["result"]
#     print(ret)
#     print(cb)
    costs = cb

costs = [f"Total Tokens: {cb.total_tokens}\n",
        f"Prompt Tokens: {cb.prompt_tokens}\n", 
        f"Completion Tokens: {cb.completion_tokens}\n", 
        f"Total Cost (USD): ${cb.total_cost}\n"]


filename = f"out_{company_name}_max_token_{MAX_TOKENS}"

with open(f"{filename}.md", "w", ) as f:
    f.write("# API Usage\n")
    f.writelines(costs)
    f.write("\n")
    
with open(f"{filename}.md", "a") as f:
    f.write(answer)