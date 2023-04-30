import os
from pprint import pprint

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

print_color = lambda text, color: print(f"\033[38;5;{color}m{text}\033[0m")

def load_documents(folder):
    filetree = lambda folder: [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
    texts = [file for file in filetree(folder) if file.endswith(".pdf")]
    return [text for file in texts for text in CharacterTextSplitter(1000, 0).split_documents(PyPDFLoader(file).load_and_split())]

def main():
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(load_documents("docs"), embeddings, persist_directory="docs.db")
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    while True:
        result = qa({"query": input("Question: ")})
        print_color(f"Answer: {result['result']}", 46)
        if os.environ["DEBUG"]:
            print_color("Source documents:", 46)
            pprint(result["source_documents"])

if __name__ == '__main__':
    main()
