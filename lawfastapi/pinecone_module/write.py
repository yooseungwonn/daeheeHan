# 임베딩, 전처리
from langchain_upstage import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

# 파인콘
from langchain_teddynote.korean import stopwords
from langchain_teddynote.community.pinecone import (
    create_sparse_encoder,
    fit_sparse_encoder,
    preprocess_documents,
    upsert_documents_parallel,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 기타
import os
from typing import List
import time

# 매크로상수 불러오기
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from app.config.static_variables import StaticVariables


# .env 로드
from dotenv import load_dotenv

load_dotenv()


def load_PDF_documents(source_uris: List[str]):
    docs = []
    for source_uri in source_uris:
        loader = PDFPlumberLoader(source_uri)
        docs.extend(loader.load())
    return docs


def create_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=StaticVariables.CHUNK_SIZE,
        chunk_overlap=StaticVariables.CHUNK_OVERLAP,
    )


def split_documents(docs, text_splitter):
    """text splitter를 사용하여 문서를 분할합니다."""
    return text_splitter.split_documents(docs)


def create_dense_embedding():
    return UpstageEmbeddings(model=StaticVariables.UPSTAGE_EMBEDDING_MODEL)


# 벡터스토어 로드. 인덱스는 미리 만들어 두는 것을 상정함
def pinecone_load_vectorstore():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(StaticVariables.PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=UpstageEmbeddings(model=StaticVariables.UPSTAGE_EMBEDDING_MODEL),
        namespace=StaticVariables.PINECONE_NAMESPACE,
    )
    return vectorstore


def pincone_hybrid_upsert(split_docs, namespace):
    # 파인콘 인덱스 로드
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc_index = pc.Index(StaticVariables.PINECONE_INDEX_NAME)

    # 문서 전처리
    contents, metadatas = preprocess_documents(
        split_docs=split_docs,
        metadata_keys=["source", "page", "total_pages"],
        use_basename=True,
    )

    # sparse 인코더 생성
    sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")
    saved_path = fit_sparse_encoder(
        sparse_encoder=sparse_encoder,
        contents=contents,
        save_path=StaticVariables.SPARSE_ENCODER_PKL_PATH,
    )

    upstage_embeddings = UpstageEmbeddings(
        model=StaticVariables.UPSTAGE_EMBEDDING_MODEL
    )

    upsert_documents_parallel(
        index=pc_index,
        namespace=namespace,
        contents=contents,
        metadatas=metadatas,
        sparse_encoder=sparse_encoder,
        embedder=upstage_embeddings,
        batch_size=32,
        max_workers=30,
    )


# 디렉토리 내 모든 파일 경로 가져오기(하위폴더 포함)
def list_files_with_paths(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def write_pinecone_with_docs(source_uri, namespace):
    sources = list_files_with_paths(source_uri)
    docs = load_PDF_documents(sources)
    text_splitter = create_text_splitter()
    split_docs = split_documents(docs, text_splitter)
    pincone_hybrid_upsert(split_docs, namespace)  # 문서 벡터DB에 저장


if __name__ == "__main__":
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("PINECONE Index가 있는 지 확인합니다(무료계정은 Index 1 개만 소유할 수 있음): ", StaticVariables.PINECONE_INDEX_NAME)
    try:
        pc.describe_index(StaticVariables.PINECONE_INDEX_NAME)
    except Exception as e:
        print("필요한 Index가 존재하지 않습니다. 새로운 Index를 생성합니다: ", StaticVariables.PINECONE_INDEX_NAME)
        pc.create_index(
            name=StaticVariables.PINECONE_INDEX_NAME,
            dimension=4096, # Replace with your model dimensions
            metric="dotproduct", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        print("Index 생성 완료!")
        
    print("PINECONE Index를 로드합니다...")
    while not pc.describe_index(StaticVariables.PINECONE_INDEX_NAME).status['ready']:
        time.sleep(1)
    index = pc.Index(StaticVariables.PINECONE_INDEX_NAME)
    print("Index 로드 완료!")
    
    print("Index 내 namespace 중복여부를 확인합니다: ", StaticVariables.PINECONE_NAMESPACE)
    existing_namespaces = index.describe_index_stats()['namespaces']
    if StaticVariables.PINECONE_NAMESPACE in existing_namespaces:
        print("중복된 namespace가 존재합니다. namespace를 삭제합니다.")
        index.delete(namespace=StaticVariables.PINECONE_NAMESPACE, delete_all=True)
        print("중복된 namespace 삭제 완료!")
        
    print("namespace에 데이터를 저장합니다.")
    write_pinecone_with_docs(
        StaticVariables.PDF_DIRECTORY_PATH, StaticVariables.PINECONE_NAMESPACE
    )
    print("데이터 저장 완료! sparse_encoder.pkl 파일은 고이 간직해주세요(원래 경로에 두세요)")