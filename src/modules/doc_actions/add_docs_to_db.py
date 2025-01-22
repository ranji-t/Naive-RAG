# Third Party Imports
from langchain.docstore.document import Document
from langchain_chroma.vectorstores import Chroma

# Internal Imports
from .doc_actions import gen_doc_id as gen_doc_id
from .doc_actions import gen_doc_ids as gen_doc_ids


def add_new_docs_to_db(docs: list[Document], chroma_store: Chroma):
    # New Never Seen Before Doc IDs
    new_document_ids = set(
        set(gen_doc_ids(docs)).difference(
            set(chroma_store.get(include=["metadatas"]).get("ids"))
        )
    )
    if not new_document_ids:
        return []

    # New Docs Only
    new_docs = [doc for doc in docs if gen_doc_id(doc) in new_document_ids]

    # Return
    return add_docs_to_db(docs=new_docs, chroma_store=chroma_store)


def add_docs_to_db(docs: list[Document], chroma_store: Chroma) -> list[str]:
    return chroma_store.add_documents(documents=docs, **{"ids": gen_doc_ids(docs)})
