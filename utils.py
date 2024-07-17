from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import yaml

with open(f"hyperparameters.yaml") as f:
    params = yaml.load(f, Loader=yaml.loader.SafeLoader)

K_RESULTS = params["k_results"]


def pdf_2_txt(pdf):

    lec = PdfReader(pdf)
    texte_brut = ''
    for i, page in enumerate(lec.pages):
        text = page.extract_text()
        if text:
            texte_brut += text

    return texte_brut


def docx_2_txt(docx: str):

    lec = Docx2txtLoader(docx)
    loader = lec.load()
    texte_brut = ""
    for i in loader:
        texte_brut += i.page_content + '\n'

    return texte_brut


def txt_splitter(text: str,
                 separator: str,
                 chunk_size: int,
                 chunk_overlap: int):

    splitted_txt = CharacterTextSplitter(separator=separator,
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap,
                                        length_function=len
                                        )

    return splitted_txt.split_text(text)


def txt_embedding(splitted_text,
                  embedding_function):

      return FAISS.from_texts(splitted_text, embedding_function)


def prompt_output(prompt,
                  embeddings,
                  model,
                  k_results=K_RESULTS):


    PROMPT_TEMPLATE = """

    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {prompt}
    """

    documents_similaires = embeddings.similarity_search(prompt, k = k_results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in documents_similaires])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_fin = prompt_template.format(context=context_text, prompt=prompt)

    # inputs = {
    #     'input_documents': context_text,
    #     'question': requete
    # }

    response = model.invoke(prompt_fin)
    # response = model.invoke(inputs)

    return response.content