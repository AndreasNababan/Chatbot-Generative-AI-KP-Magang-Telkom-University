import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Chatbot KP & Magang",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# LOAD LLM
# =====================================

@st.cache_resource
def load_llm():
    llm = OllamaLLM(
        model="llama3.2"
    )
    return llm

try:
    llm = load_llm()
except Exception as e:
    st.error(f"Ollama tidak dapat dijalankan: {e}")
    st.stop()

# =====================================
# LOAD DATABASE
# =====================================

@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )
    database = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return database


db = load_database()

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)

    st.markdown("### KP & Magang AI")

    if st.button("🗑️ Chat Baru", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
# =====================================
# HEADER
# =====================================

st.markdown("""
<div style='text-align:center;margin-top:20px;margin-bottom:30px'>
<h1 style='font-size:42px'>🎓 KP & Magang AI</h1>
<p style='color:#94a3b8'>
Asisten Informasi Kerja Praktik dan Magang Telkom University
</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.stApp{
    background:#0b1120;
}

.user-msg{
    background:#2563eb;
    color:white;
    padding:14px;
    border-radius:18px;
    margin-bottom:10px;
}

.bot-msg{
    background:#111827;
    border:1px solid #1f2937;
    color:white;
    padding:14px;
    border-radius:18px;
    margin-bottom:10px;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container{
    max-width:900px;
    margin:auto;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================================
# INPUT
# =====================================

prompt = st.chat_input("Tanyakan tentang KP atau Magang...")

if "quick_question" in st.session_state:
    prompt = st.session_state.pop("quick_question")

def extract_answer(text):
    try:
        if "Jawaban:" in text:
            answer = text.split("Jawaban:")[1]

            if "Kategori:" in answer:
                answer = answer.split("Kategori:")[0]

            return answer.strip()

        return text.strip()

    except:
        return text.strip()

# =====================================
# CHAT PROCESS
# =====================================

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Mencari jawaban..."):
        try:
            query = f"query: {prompt}"
            docs = db.similarity_search(
                prompt,
                k=5
            )

            context = ""
            for doc in docs:
                context += doc.page_content + "\n\n"

            if docs:
                rag_prompt = f"""
Anda adalah chatbot resmi KP dan Magang Telkom University.

ATURAN WAJIB:

1. Jawab HANYA berdasarkan konteks.
2. Jangan menulis:
   - "Menurut konteks"
   - "Berdasarkan konteks"
   - "Pertanyaan:"
   - "Jawaban:"
   - "Berikut jawaban"
3. Langsung berikan jawaban akhir.
4. Maksimal 3 kalimat.
5. Jika tidak ada informasi jawab:

Maaf informasi tersebut tidak ditemukan dalam pedoman KP dan Magang.

KONTEKS:
{context}

PERTANYAAN:
{prompt}

JAWABAN:
"""

                answer = llm.invoke(rag_prompt)
            else:
                answer = "Maaf, informasi tidak ditemukan."

        except Exception as e:
            answer = f"Terjadi kesalahan: {str(e)}"

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Generative AI Project • Telkom University • 2026"
)
