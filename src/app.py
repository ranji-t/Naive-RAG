import marimo

__generated_with = "0.10.15"
app = marimo.App(width="full", app_title="RAG-Application")


@app.cell
def marimo_init():
    # Third party imports
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Third Pary Imports
    from modules import get_config, get_ollama_embedder, get_chroma_store
    from modules.doc_actions import load_docs, add_new_docs_to_db
    # from modules.doc_actions import

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

    # Get Sub Samples
    doc_samples = docs[:500]

    # Add Documsnts to DB
    id_of_new_docs = add_new_docs_to_db(doc_samples, chroma_store)

    # Print data
    print(f"No of new documents = {len(id_of_new_docs)}")

    # Display data
    id_of_new_docs
    return (
        add_new_docs_to_db,
        chroma_store,
        config,
        doc_samples,
        docs,
        get_chroma_store,
        get_config,
        get_ollama_embedder,
        id_of_new_docs,
        load_docs,
    )


if __name__ == "__main__":
    app.run()
