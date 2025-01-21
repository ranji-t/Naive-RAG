import marimo

__generated_with = "0.10.15"
app = marimo.App(width="full", app_title="RAG-Application")


@app.cell
def marimo_init():
    # Third party imports
    import marimo as mo
    return (mo,)


@app.cell
def data_ingestion():
    # Third Pary Imports
    from modules import get_config, get_ollama_embedder, get_chroma_store
    from modules.doc_actions import load_docs, add_new_docs_to_db


    # Congiguration Import
    config = get_config("config.toml")

    # Set up Chroma DB with Embedder
    chroma_store = get_chroma_store(
        embedding_function=get_ollama_embedder(config.get("embedder").get("name")),
        chroma_config=config.get("chroma-client"),
        collection_name=config.get("chroma-collection").get("name"),
    )

    # Process Documents
    docs = load_docs(
        dir_path=config.get("docs").get("glob_pattern"),
        **config.get("docs").get("splitter"),
    )


    # Add Documsnts to DB
    id_of_new_docs = add_new_docs_to_db(docs, chroma_store)

    # Print data
    print(f"No of new documents = {len(id_of_new_docs)}")

    # Display data
    id_of_new_docs
    return (
        add_new_docs_to_db,
        chroma_store,
        config,
        docs,
        get_chroma_store,
        get_config,
        get_ollama_embedder,
        id_of_new_docs,
        load_docs,
    )


@app.cell
def _():
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate

    # Templates
    template = """Question: {question}

    Answer: Let's think step by step."""
    # Create Prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Create Modle
    model = OllamaLLM(model="deepseek-r1:1.5b")

    # Create Chain Prompt + Model
    chain = prompt | model
    return ChatPromptTemplate, OllamaLLM, chain, model, prompt, template


@app.cell
def _(chain):
    chain.invoke({"question": "What is LangChain?"})
    return


if __name__ == "__main__":
    app.run()
