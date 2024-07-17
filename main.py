from utils import pdf_2_txt, docx_2_txt, txt_splitter, txt_embedding


def rag_document_embeddings(doc,
                            separator,
                            chunk_size,
                            chunk_overlap,
                            embedding_function,
                            type_file):

    if type_file == "pdf":
        text_brut = pdf_2_txt(doc)

    elif type_file == "docx":
        text_brut = docx_2_txt(doc)

    splitted_txt = txt_splitter(text=text_brut,
                                separator=separator,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)

    embeddings = txt_embedding(splitted_txt,
                               embedding_function)

    return embeddings
