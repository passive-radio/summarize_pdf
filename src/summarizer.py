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
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
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
            self.api_cost = [
                f"Total Tokens: {cb.total_tokens}\n",
                f"Prompt Tokens: {cb.prompt_tokens}\n",
                f"Completion Tokens: {cb.completion_tokens}\n",
                f"Total Cost (USD): ${cb.total_cost}\n\n",
            ]
        
        return self._cb
            
    def out(self, filepath):
        with open(filepath, "w") as f:
            f.writelines(self.api_cost)

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
        self.texts = self.splitter.split_documents([self.pages[0]])
        self.embeddings = OpenAIEmbeddings()
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
        
        qa_prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa_chain = LLMChain(
            llm=self.llm,
            prompt=qa_prompt
        )

        with get_openai_callback() as cb:
            
            self._answer = qa_chain.run({"context": self.pages[0], "question": qa_document.ir_header})
            self._answer += "\n\n"
            self._cb = cb
            self._api_cost = {"total_tokens": cb.total_tokens, "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens, "total_cost(USD)": cb.total_cost}

            print(self._answer)
            
        self.texts = self.splitter.split_documents(self.pages[1:])
        self.vectordb = Chroma.from_documents(self.texts, self.embeddings)
        
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectordb.as_retriever(),
        )
        
        with get_openai_callback() as cb:
            
            self._answer += qa.run(qa_document.ir_content)
            api_cost = {"total_tokens": cb.total_tokens, "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens, "total_cost(USD)": cb.total_cost}
            self._api_cost = {k: self._api_cost.get(k, 0) + api_cost.get(k, 0) for k in set(self._api_cost) & set(api_cost)}
                    
        self.api_cost = [
                f"Total Tokens: {self._api_cost['total_tokens']}\n",
                f"Prompt Tokens: {self._api_cost['prompt_tokens']}\n",
                f"Completion Tokens: {self._api_cost['completion_tokens']}\n",
                f"Total Cost (USD): ${self._api_cost['total_cost(USD)']}\n\n",
            ]
        
        return self._cb
    
if __name__ == "__main__":
    summarizer = IRSummarizer("gpt-3.5-turbo", 1800)
    summarizer.load_document("../asset/ir_japanhospicehld.pdf")
    summarizer.run()
    summarizer.out("../out/ir_japanhospicehld_summary_03.md")