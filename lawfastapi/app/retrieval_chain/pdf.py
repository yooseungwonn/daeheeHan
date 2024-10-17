from retrieval_chain.base import RetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from config.static_variables import StaticVariables


class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri):
        self.source_uri = source_uri

    def load_documents(self, source_uris: List[str]):
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())

        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=StaticVariables.CHUNK_SIZE, chunk_overlap=StaticVariables.CHUNK_OVERLAP)


    # def create_text_splitter(self):
    #     return SemanticChunker()