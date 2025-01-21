import marimo

__generated_with = "0.10.15"
app = marimo.App(width="full", app_title="RAG-Application")


@app.cell
def _():
    # Third party imports
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Third Pary Imports
    from modules import get_config, get_ollama_embedder, get_chroma_store
    from modules.doc_actions import load_docs, add_docs_to_db

    # Congiguration Import
    config = get_config("config.toml")

    # Set up Ollama Embedder
    embedding_function = get_ollama_embedder(config)

    # Set up Chroma DB with Embedder
    chroma_store = get_chroma_store(
        embedding_function=embedding_function,
        chroma_config=config.get("chroma-client"),
        collection_name=config.get("chroma-collection").get("name"),
    )

    # Process Documents
    docs = load_docs(
        dir_path=config.get("docs").get("glob_pattern"),
        **config.get("docs").get("splitter"),
    )

    # Get Sub Samples
    doc_samples = docs[:20]

    # Add Sample Docs to Chroma DB
    add_docs_to_db(docs=doc_samples, chroma_store=chroma_store)
    return (
        add_docs_to_db,
        chroma_store,
        config,
        doc_samples,
        docs,
        embedding_function,
        get_chroma_store,
        get_config,
        get_ollama_embedder,
        load_docs,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
