import os
import platform

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from secret import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0.0, max_tokens=2500)

filepath = "../asset/lecture.pdf"
out_dir = "../out"
out_filename = f"{out_dir}/summary_02.md"

query = """与えられた生物学に関する講義のレジュメを要約し以下のフォーマットを必ず守り英語で出力してください。```
# 概要
    1. author
    2. institution
    3. summary

# 遺伝子重複の種類
    (more than 7 topics, more than 60 words on each topic)
    1. a highlight of this section (eg. theory, experimental result)
    2.
    3.
    4.
    5.
    6.
    7.

# 遺伝子重複後の運命
    (more than 7 topics, more than 60 words on each topic)
    1. a highlight of this section (eg. theory, experimental result)
    2. 
    3. 

# アミノ酸配列に対する自然淘汰圧の推定
    (more than 7 topics, more than 60 words on each topic)
    1. a highlight of this section (eg. theory, experimental result)
    2. 
    3. 
```"""

loader = PyPDFLoader(filepath)
pages = loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n", "\n\n"],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    )

texts = splitter.split_documents(pages[:54])
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    )



answer = ""
usage = []
with get_openai_callback() as cb:
    answer = qa.run(query)
    usage = [f"Total Tokens: {cb.total_tokens}\n",
             f"Prompt Tokens: {cb.prompt_tokens}\n",
             f"Completion Tokens: {cb.completion_tokens}\n",
             f"Total Cost (USD): ${cb.total_cost}\n\n",
            ]
    print(answer)

with open(out_filename, "w") as f:
    f.writelines(usage)

with open(out_filename, "a") as f:
    f.write(answer)
