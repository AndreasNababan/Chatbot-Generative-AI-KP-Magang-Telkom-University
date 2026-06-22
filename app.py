import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


st.set_page_config(
    page_title="Chatbot KP & Magang",
    page_icon="🎓",
    layout="wide"
)


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


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)

    st.markdown("### KP & Magang AI")

    if st.button("🗑️ Chat Baru", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Tanyakan tentang KP atau Magang...")

# Pesan penolakan standar untuk pertanyaan di luar topik
PESAN_DILUAR_TOPIK = (
    "Maaf, saya hanya dapat menjawab pertanyaan seputar **Kerja Praktik (KP)** "
    "dan **Magang** di Telkom University. "
    "Silakan ajukan pertanyaan yang berkaitan dengan topik tersebut. 😊"
)

# Batas skor kemiripan (semakin kecil = semakin mirip di Chroma cosine distance)
SIMILARITY_THRESHOLD = 1.2


def is_related_to_kp_magang(text):
    """Cek apakah pertanyaan mengandung kata kunci terkait KP/Magang."""
    keywords = [
        # Topik utama
        "kp", "magang", "kerja praktik", "kerja praktek", "internship",
        "praktek kerja", "praktik kerja", "pkl", "praktik industri",
        # Proses & administrasi
        "pendaftaran", "daftar", "registrasi", "persyaratan", "syarat",
        "prosedur", "tahap", "langkah", "alur", "proses",
        "formulir", "form", "dokumen", "berkas", "surat",
        "proposal", "laporan", "sidang", "seminar", "presentasi",
        "nilai", "penilaian", "bimbingan", "pembimbing", "dosen",
        "koordinator", "supervisor", "mentor",
        # Waktu & jadwal
        "jadwal", "periode", "semester", "durasi", "waktu",
        "deadline", "batas waktu", "tenggat",
        # Lokasi & perusahaan
        "perusahaan", "instansi", "mitra", "lokasi", "tempat",
        "penempatan", "posisi",
        # SKS & akademik
        "sks", "mata kuliah", "kurikulum", "transkrip",
        "konversi", "kredit",
        # Umum terkait
        "telkom university", "tel-u", "telu",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def extract_answer(text):
    """Ekstrak jawaban bersih dari output LLM."""
    try:
        if "Jawaban:" in text:
            answer = text.split("Jawaban:")[1]
            if "Kategori:" in answer:
                answer = answer.split("Kategori:")[0]
            return answer.strip()
        return text.strip()
    except Exception:
        return text.strip()


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
            # ===== LAYER 1: Filter kata kunci =====
            if not is_related_to_kp_magang(prompt):
                answer = PESAN_DILUAR_TOPIK

            else:
                # ===== LAYER 2: Cek relevansi dari database =====
                docs = db.similarity_search_with_score(
                    prompt,
                    k=5
                )

                # Filter dokumen yang relevan berdasarkan skor
                relevant_docs = [
                    (doc, score) for doc, score in docs
                    if score <= SIMILARITY_THRESHOLD
                ]

                if not relevant_docs:
                    # Tidak ada dokumen relevan di database
                    answer = PESAN_DILUAR_TOPIK

                else:
                    # Bangun konteks dari dokumen relevan saja
                    context = ""
                    for doc, score in relevant_docs:
                        context += doc.page_content + "\n\n"

                    # ===== LAYER 3: Prompt ketat ke LLM =====
                    rag_prompt = f"""[SYSTEM]
Anda adalah chatbot resmi Kerja Praktik (KP) dan Magang di Telkom University.

ATURAN KETAT YANG TIDAK BOLEH DILANGGAR:
1. Anda HANYA boleh menjawab pertanyaan tentang KP dan Magang Telkom University.
2. Anda HANYA boleh menggunakan informasi dari KONTEKS di bawah. DILARANG menggunakan pengetahuan umum atau informasi dari luar konteks.
3. Jika pertanyaan TIDAK berkaitan dengan KP atau Magang, Anda WAJIB menjawab PERSIS: "Maaf, saya hanya dapat menjawab pertanyaan seputar Kerja Praktik (KP) dan Magang di Telkom University."
4. Jika KONTEKS tidak mengandung jawaban, Anda WAJIB menjawab PERSIS: "Maaf, informasi tersebut belum tersedia dalam database saya."
5. DILARANG menjawab pertanyaan tentang topik lain seperti: cuaca, matematika, coding, berita, politik, hiburan, atau topik umum lainnya.
6. DILARANG mengarang atau menambahkan informasi yang tidak ada dalam KONTEKS.
7. Jawab dalam Bahasa Indonesia yang baik dan sopan.

KONTEKS:
{context}

[USER]
{prompt}

[ASSISTANT]
"""

                    raw_answer = llm.invoke(rag_prompt)
                    answer = extract_answer(raw_answer)

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


st.divider()

st.caption(
    "Generative AI Project • Telkom University • 2026"
)
