# Standard Imports
from glob import iglob
from itertools import chain
from hashlib import sha256

# Third Party Imports
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_docs(
    dir_path: str, chunk_size: int = 4000, chunk_overlap: int = 200
) -> list[Document]:
    return list(
        chain(
            *(
                TextLoader(file_path=file_path).load_and_split(
                    text_splitter=RecursiveCharacterTextSplitter(
                        separators=None,
                        keep_separator=True,
                        is_separator_regex=False,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        add_start_index=True,
                        strip_whitespace=True,
                    )
                )
                for file_path in iglob(dir_path)
            )
        )
    )


def gen_doc_ids(docs: list[Document]) -> list[str]:
    return [get_doc_id(doc) for doc in docs]


def get_doc_id(doc: Document) -> str:
    return sha256(
        f"{doc.page_content}--{doc.metadata['source']}--{doc.metadata['start_index']}".encode()
    ).hexdigest()
