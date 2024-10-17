def trim_path(path):
    return path.lstrip('./').rsplit('.', 1)[0]


def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{trim_path((doc.metadata['source']))}</source></document>"
            for doc in docs
        ]
    )

### 백업용 ###
# def format_docs(docs):
#     return "\n".join(
#         [
#             f"<document><content>{doc.page_content}</content><source>{(doc.metadata['source'])}</source><page>{int(doc.metadata['page'])+1}</page></document>"
#             for doc in docs
#         ]
#     )

# 검색 결과 문서화
# def format_searched_docs(docs):
#     return "\n".join(
#         [
#             f"<document><content>{doc['content']}</content><source>{doc['url']}</source></document>"
#             for doc in docs
#         ]
#     )
