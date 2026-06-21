from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class Chatbot:

    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embedding
        )

    def ask(self, question):

        docs = self.db.similarity_search(
            question,
            k=3
        )

        if len(docs) == 0:
            return "Maaf, informasi tidak ditemukan."

        return docs[0].page_content