import os
import shutil
import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================
# LOAD CSV
# =====================================

CSV_PATH = "data/qa_kp_magang.csv"

df = pd.read_csv(CSV_PATH)

required_columns = ["question", "answer", "category"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan pada CSV")

# =====================================
# HAPUS DATABASE LAMA
# =====================================

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

# =====================================
# BUILD DOCUMENTS
# =====================================

documents = []
for _, row in df.iterrows():
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    category = str(row["category"]).strip()

    content = f"Pertanyaan: {question}\nJawaban: {answer}\nKategori: {category}"

    documents.append(
        Document(
            page_content=content,
            metadata={
                "question": question,
                "category": category
            }
        )
    )

# =====================================
# EMBEDDING MODEL
# =====================================

embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# =====================================
# CREATE CHROMA DB
# =====================================

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="chroma_db"
)

vectordb.persist()

print("Berhasil membuat database.")
print(f"Total data: {len(documents)}")
