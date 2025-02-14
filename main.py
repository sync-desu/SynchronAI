"""
Tested on Python version: 3.10.11
Run command `pip install -r requirements.txt` to install all dependencies.
Ensure you have Ollama and its servive running.
    - Pull the following models on Ollama:
        -- nomic-embed-text:latest (for embedding)
        -- qwen2-math:7b (for scientific reasoning)
"""


import os

from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever # for typehinting
from langchain.chains import RetrievalQA # retrival chain used for RAG
from langchain_chroma import Chroma # chromadb to store Vector Embeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM # ollama embedding and large language model




class SynchronAI:
    def __init__(self, *, datastore_path: str = "data.jsonl", vectorstore_path: str = "./chroma_db",
                 force_vectorstore_overwrite: bool = False) -> None:
        """
        Storing an instance vectorstore variable which can be accessible anywhere inside the instance.
        Primarily to reduce reuse of `self.__create_vectorstore` method.
        """
        self.__ollama_embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.__ollama_model = OllamaLLM(model="qwen2-math:7b", temperature=0.7)
        self.__vectorstore = self.__create_vectorstore(
            datastore_path=datastore_path,
            vectorstore_path=vectorstore_path,
            force_overwrite=force_vectorstore_overwrite
        )

    @staticmethod
    def __ingest_data(datastore_path: str) -> list:
        """
        Extracting JSONL data and converting each line into a traditional Python Dictionary.
        Adding an extra "question_number" field which is dependent on the line-number of the data.
        Formatting and returning formatted data lines as documents to store as vector embeds.
            - Entire process is called Ingestion.
        """
        data = []
        with open(file=datastore_path, mode="r", encoding="utf-8") as f:
            for x, line in enumerate(iterable=f, start=1):
                json_data = eval(line.strip()) # eval or json.loads, since each line is a dictionary
                json_data["question_number"] = x # adding a question_number field
                data.append(json_data)
        return [(f"The Question Number for this question is '{line['question_number']}', "
                f"and the Subject Name is '{line['subject']}'. "
                f"The The Question is stated to be '{line['question']}', "
                f"and the Options are '{', '.join(line['options'])}'. "
                f"The Answer for this question is '{line['answer']}'.")
            for line in data]
    
    @staticmethod
    def __build_chain(ollama_model: OllamaLLM, retriever: VectorStoreRetriever, prompt_template: PromptTemplate) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=ollama_model,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )

    @property
    def __prompt_template(self) -> PromptTemplate:
        """
        A simple prompt template, guiding the LLM into generating a curated response.
        """
        system_message = (
            "Answer the following question based only on the provided context.\n"
            "You are an Assistant Scientist with a broad knowledge about Physics, Chemistry and Mathematics.\n"
            "You must analyze the question and be able think step-by-step and provide a detailed answer.\n"
            "You will gain $1,000 if you answer only when the context is sufficient and related to the question.\n"
            "You will lose $2,000 if you answer when there is insufficient context.\n"
            "Do not mention your gain/loss of money to the user.\n"
            "Do not mention anything about the context to the user.\n"
            "Do not respond with empty lines, say you do not know if you can not respond.\n"
            "Your ultimate goal is to maximize the money you gain.\n"
        )
        template = (
            f"{system_message}\n\n"
            "Context - {context}\n\n"    # (relevant documents) injected by langchain when chain is invoked
            "Question - {question}\n\n"  # (query) also injected by langchain when chain is invoked
            "Answer - ..."
        )
        return PromptTemplate(input_variables=["context", "question"], template=template)

    def __create_vectorstore(self, datastore_path: str, vectorstore_path: str, force_overwrite: bool) -> Chroma:
        """
        Creates or Returns a Chroma vectorstore.
            - Create: Uses the OllamaEmbeddings model (nomic-embed-text) to create
                vector embedings, and creates a persistent ChromaDB vectorstore
                to store the vector embeddings.
            - Return: Returns the created or existing ChromaDB vectorstore.
        """
        if os.path.exists(vectorstore_path):
            if not force_overwrite:
                print("Fetching the existing ChromaDB vectorstore...")
                return Chroma(embedding_function=self.__ollama_embedder, persist_directory=vectorstore_path)
        print(("Creating a new" if not force_overwrite else "Overwriting the existing") + " ChromaDB vectorstore...")
        documents = self.__ingest_data(datastore_path)
        vectorstore = Chroma.from_texts(documents, self.__ollama_embedder, persist_directory=vectorstore_path)
        return vectorstore

    def ask(self, query: str, *, use_llm: bool = False, relevancy_threshold: float = 0.5) -> None:
        """
        Query data from the ChromaDB vectorstore based on the similarity of the data.
            - When 'use_llm' is set to True, use a Retrieval-Augmented Generation technique to
                ensure the LLM generates relevant information to the query.
        If the relevancy score of the first searched document does not hit the relevancy_threshold,
            there is no further processing and a system error message is sent.
        """
        relevancy_score = self.__vectorstore.similarity_search_with_relevance_scores(query, k=1)[0][1] # parsing the score
        if relevancy_score < relevancy_threshold:
            return "SYSTEM: This query is not irrelevant to any of the data we possess."
        if not use_llm:
            retrieved_docs = self.__vectorstore.similarity_search(query, k=2) # retrieve documents using similarity search
            for doc in retrieved_docs:
                return doc.page_content  # return k top-most retrieved relevant documents
        chain = self.__build_chain(
            self.__ollama_model,
            self.__vectorstore.as_retriever(),
            self.__prompt_template    # the template receives the retrieved documents as {context}
        )                             # and the question as {question} when chain.invoke() is called.
        response = chain.invoke(query)
        return response["result"].strip()  # post-processing




if __name__ == "__main__":
    chatbot = SynchronAI(force_vectorstore_overwrite=True)
    response1 = chatbot.ask("What is a bicycle?", use_llm=True)
    print(f"Response 1: {response1}")
    response2 = chatbot.ask("Gas evolved when Zinc reacts with Hydrochloric Acid",  use_llm=True)
    print(f"Response 2: {response2}")
    response3 = chatbot.ask("Do you like kittens?", use_llm=True)
    print(f"Response 3: {response3}")
    response4 = chatbot.ask("What is the SI unit of work?", use_llm=True)
    print(f"Response 4: {response4}")
