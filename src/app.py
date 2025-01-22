import marimo

__generated_with = "0.10.15"
app = marimo.App(width="full", app_title="RAG-Application")


@app.cell
def marimo_init():
    # Third party imports
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# **Data Ingestion**""")
    return


@app.cell
def data_ingestion(mo):
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
    mo.md(f"No of new documents = {len(id_of_new_docs)}")
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
def _(mo):
    mo.md("""# **Chain of Thought**""")
    return


@app.cell
def _(mo):
    # Select  Model
    llm_modle_name = mo.ui.dropdown(
        options=["deepseek-r1:1.5b", "deepseek-r1:7b"], value="deepseek-r1:1.5b"
    )

    # Display Dropdown  Options
    llm_modle_name
    return (llm_modle_name,)


@app.cell
def _():
    # Third Party Imports
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM


    # Templates
    template = """Question: {question}

    System: Write the outputs Markdown

    Answer: Let's think step by step"""
    # Create Prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Create Modle
    model = OllamaLLM(model="deepseek-r1:1.5b")

    # Create Chain Prompt + Model
    chain = prompt | model
    return ChatPromptTemplate, OllamaLLM, chain, model, prompt, template


@app.cell
def _(mo):
    # Get data from user
    text_input = mo.ui.text(
        value="",
        placeholder="Write ur question here...",
        full_width=True,
    )

    # Show data
    text_input
    return (text_input,)


@app.cell
def _(chain, mo, text_input):
    # Text input
    if text_input.value == "":
        output = ""
    else:
        # Get output
        output = chain.invoke({"question": text_input.value})

    # Show Output of the LLM
    mo.md(output)
    return (output,)


@app.cell
def _(mo):
    mo.md(r"""# **Retrival of Query**""")
    return


@app.cell
def _(mo):
    # Get data from user
    query_input = mo.ui.text(
        value="",
        placeholder="Write ur question here...",
        full_width=True,
    )

    # Show data
    query_input
    return (query_input,)


@app.cell
def _(chroma_store, query_input):
    chroma_store.search(
        query=query_input.value,
        search_type="similarity",
        **{
            "k": 7,
            # "score_threshold": 0.15,
            # "fetch_k": 3,
            # "lambda_mult": 0.5
        },
    )
    return


if __name__ == "__main__":
    app.run()
