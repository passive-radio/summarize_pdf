import os
import platform
from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod

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

from recipe import qa_document


filepath = "../asset/lecture.pdf"
out_dir = "../out"
out_filename = f"{out_dir}/summary_02.md"


class Summarizer(metaclass=ABCMeta):
    def __init__(self, model_name: str, max_tokens: int = None) -> None:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(model_name = model_name, temperature=0.0, max_tokens=max_tokens)
    
    def load_document(self, filepath: str) -> None:
        self.loader = PyPDFLoader(filepath)
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "\n\n"],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    @abstractmethod
    def run(self, chain_type: str = "stuff") -> dict:
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectordb.as_retriever(),
        )

        self._answer = ""
        self._api_cost = []
        with get_openai_callback() as cb:
            self._answer = qa.run(qa_document.general)
            self._cb = cb
            self._api_cost = [
                f"Total Tokens: {cb.total_tokens}\n",
                f"Prompt Tokens: {cb.prompt_tokens}\n",
                f"Completion Tokens: {cb.completion_tokens}\n",
                f"Total Cost (USD): ${cb.total_cost}\n\n",
            ]
        
        return self._cb
            
    def out(self, filepath):
        with open(filepath, "w") as f:
            f.writelines(self._api_cost)

        with open(filepath, "a") as f:
            f.write(self._answer)
    
    @staticmethod
    def get_answer(self):
        return self._answer
    
    @staticmethod
    def get_api_cost(self):
        return self._api_cost
    
class IRSummarizer(Summarizer):
    def __init__(self, model_name: str, max_tokens: int = None) -> None:
        super().__init__(model_name, max_tokens)
        
    def run(self, chain_type: str = "stuff"):
        self.pages = self.loader.load_and_split()
        print(self.pages[0].page_content)
        self.texts = self.splitter.split_documents([self.pages[0]])
        self.embeddings = OpenAIEmbeddings()
        print(self.texts)
        self.vectordb = Chroma.from_documents(self.texts, self.embeddings)
        
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectordb.as_retriever(),
        )

        self._answer = ""
        self._api_cost = []
        
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        with get_openai_callback() as cb:
            
            
            
            
            self._answer = qa.run(qa_document.ir_header)
            self._cb = cb
            self._api_cost = [
                f"Total Tokens: {cb.total_tokens}\n",
                f"Prompt Tokens: {cb.prompt_tokens}\n",
                f"Completion Tokens: {cb.completion_tokens}\n",
                f"Total Cost (USD): ${cb.total_cost}\n\n",
            ]
            
        self.texts = self.splitter.split_documents(self.pages[1:])
        self.vectordb = Chroma.from_documents(self.texts, self.embeddings)
        
        with get_openai_callback() as cb:
            self._answer += qa.run(qa_document.ir_content)
            self._cb = {k: self._cb.get(k, 0) + cb.get(k, 0) for k in set(self._cb) & set(cb)}
            self._api_cost = [
                f"Total Tokens: {self._cb.total_tokens}\n",
                f"Prompt Tokens: {self._cb.prompt_tokens}\n",
                f"Completion Tokens: {self._cb.completion_tokens}\n",
                f"Total Cost (USD): ${self._cb.total_cost}\n\n",
            ]
        
        return self._cb
    
if __name__ == "__main__":
    summarizer = IRSummarizer("gpt-3.5-turbo", 2000)
    summarizer.load_document("../asset/ir_dena.pdf")
    summarizer.run()
    summarizer.out("../out/ir_dena_summary_02.md")