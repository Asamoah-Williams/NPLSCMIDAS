from langchain_community.document_loaders import PyPDFLoader
from pyprojroot import here
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import yaml
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


class PrepareVectorDB:
    def __init__(self, 
                 doc_dir, 
                 chunk_size, 
                 chunk_overlap,
                 embedding_model,
                 docs,
                 collection_name
                 ):
        
        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.docs = docs
        self.collection_name = collection_name

    def path_maker(self, file_name, doc_dir):
        return os.path.join(here(doc_dir), file_name)
    
    def run(self):
        file_list = os.listdir(here(self.docs))
        docs = [PyPDFLoader(self.path_maker(fn, self.doc_dir)).load_and_split() for fn in file_list]
        docs = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        doc_splits = text_splitter.split_documents(docs)
        uuids = [str(uuid4()) for _ in range(len(doc_splits))]
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        vector_store.add_documents(documents=doc_splits, ids=uuids)
        

if __name__ == "__main__":
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")

    with open(here("src/configs/tools_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    chunk_size = app_config["guiderag_configs"]["chunk_size"] 
    chunk_overlap = app_config["guiderag_configs"]["chunk_overlap"]
    embedding_model = app_config["guiderag_configs"]["embedding_model"]
    docs = app_config["guiderag_configs"]["docs"]
    collection_name = app_config["guiderag_configs"]["collection_name"]
    doc_dir = app_config["guiderag_configs"]["docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        docs=docs,
        collection_name=collection_name)

    prepare_db_instance.run()
