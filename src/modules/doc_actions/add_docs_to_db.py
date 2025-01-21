# Third Party Imports
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

# Internal Imports
from .doc_actions import gen_doc_ids as gen_doc_ids


def add_docs_to_db(docs: list[Document], chroma_store: Chroma) -> list[str]:
    return chroma_store.add_documents(documents=docs, **{"ids": gen_doc_ids(docs)})
