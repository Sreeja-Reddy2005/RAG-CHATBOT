import streamlit as st
import base64
from openai import OpenAI
import time
import logging
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from auth import login, register
from chat_db import *

from PIL import Image
import io


logging.basicConfig(level=logging.INFO, format="%(message)s")

def log_step(name, start, end, status="SUCCESS", error=None):
    log_data = {
        "step": name,
        "status": status,
        "start_time": round(start, 4),
        "end_time": round(end, 4),
        "duration_ms": round((end - start) * 1000, 2)
    }
    if error:
        log_data["error"] = str(error)

    logging.info(json.dumps(log_data))


def rerank_documents(query, docs):
    scored_docs = []

    for d in docs:
        score = d.page_content.lower().count(query.lower())
        scored_docs.append((score, d))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return [d for _, d in scored_docs[:2]]


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="my_token"
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"
FALLBACK = "meta-llama/Llama-3.1-8B-Instruct"

create_tables()


if "chat" not in st.session_state:
    st.session_state.chat = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "use_image_next" not in st.session_state:
    st.session_state.use_image_next = False

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


def optimize_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((512, 512))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_document(file):
    file_bytes = file.read()
    file_name = file.name

    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = create_conversation(
            st.session_state.user_id,
            "Document Chat"
        )

    save_document(st.session_state.conversation_id, file_bytes)

    temp_path = f"temp_{file_name}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        if file_name.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(temp_path)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(temp_path, encoding="utf-8")

        start = time.time()
        try:
            docs = loader.load()
            end = time.time()
            log_step("DOC_LOADING", start, end, "SUCCESS")
        except Exception as e:
            end = time.time()
            log_step("DOC_LOADING", start, end, "FAILED", e)
            st.error(f"Doc error: {str(e)}")
            return

    except Exception as e:
        st.error(f"Doc error: {str(e)}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    st.session_state.vectorstore = FAISS.from_documents(
        splitter.split_documents(docs),
        embedding_model
    )

    st.success("Document ready!")


def chat_llm(prompt):
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt[:4000]}],
            
        )
        return res.choices[0].message.content
    except:
        res = client.chat.completions.create(
            model=FALLBACK,
            messages=[{"role": "user", "content": prompt[:3000]}],
        )
        return res.choices[0].message.content


def generate_summary():
    text = ""
    for m in st.session_state.chat:
        text += f"{m['role']}: {m['content']}\n"

    return chat_llm(f"Summarize:\n{text}")


if "user_id" not in st.session_state:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("Multimodal RAG Chatbot")

        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login(u, p)
            if user:
                st.session_state.user_id = user.id
                st.rerun()

else:

    st.sidebar.title("Chats")

    if st.sidebar.button("New Chat"):
        st.session_state.chat = []
        st.session_state.conversation_id = None
        st.session_state.vectorstore = None
        st.rerun()

    for cid, title in get_conversations(st.session_state.user_id):
        if st.sidebar.button(title, key=f"chat_{cid}"):

            st.session_state.conversation_id = cid
            st.session_state.chat = get_messages(cid)
            st.rerun()

    img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if img:
        st.session_state.uploaded_image = optimize_image(img.read())
        st.session_state.use_image_next = True
        st.success("Image ready")

    doc = st.file_uploader("Upload Doc", type=["pdf","txt"])

    if doc:
        process_document(doc)

    col1, col2 = st.columns([6,1])
    with col1:
        st.markdown("### 💬 Chat")
    with col2:
        if st.button("📝"):
            summary = generate_summary()
            save_summary(st.session_state.conversation_id, summary)
            st.session_state.chat.append({"role":"assistant","content":summary})

    prompt = st.chat_input("Ask...")

    if prompt:

        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(
                st.session_state.user_id,
                prompt[:50]
            )

        rag_context = ""

        if st.session_state.vectorstore:

            start = time.time()
            try:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=2)
                end = time.time()
                log_step("RETRIEVAL", start, end, "SUCCESS")
            except Exception as e:
                end = time.time()
                log_step("RETRIEVAL", start, end, "FAILED", e)
                docs = []

            start = time.time()
            try:
                docs = rerank_documents(prompt, docs)
                end = time.time()
                log_step("RERANK", start, end, "SUCCESS")
            except Exception as e:
                end = time.time()
                log_step("RERANK", start, end, "FAILED", e)

            rag_context = "\n".join([d.page_content for d in docs])

        else:
            print("[RETRIEVAL] skipped (no docs)")

    
        start = time.time()
        try:
            if st.session_state.vectorstore:
                final_prompt = f"""
Use the context to answer accurately.

{rag_context}

Question: {prompt}
"""
            else:
                final_prompt = f"""
Answer clearly and accurately.

Question: {prompt}
"""

            end = time.time()
            log_step("PROMPT_CREATION", start, end, "SUCCESS")

        except Exception as e:
            end = time.time()
            log_step("PROMPT_CREATION", start, end, "FAILED", e)

        if st.session_state.use_image_next and st.session_state.uploaded_image:

            img_base64 = base64.b64encode(st.session_state.uploaded_image).decode()

            start = time.time()
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt[:3000]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }],
                    
                )
                res = response.choices[0].message.content
                end = time.time()
                log_step("LLM_GENERATION", start, end, "SUCCESS")
            except Exception as e:
                end = time.time()
                log_step("LLM_GENERATION", start, end, "FAILED", e)
                res = "Error generating response"

            img_data = img_base64
            st.session_state.use_image_next = False
            st.session_state.uploaded_image = None

        else:

            start = time.time()
            try:
                res = chat_llm(final_prompt)
                end = time.time()
                log_step("LLM_GENERATION", start, end, "SUCCESS")
            except Exception as e:
                end = time.time()
                log_step("LLM_GENERATION", start, end, "FAILED", e)
                res = "Error generating response"

            img_data = None

        start = time.time()
        try:
            save_message(st.session_state.conversation_id, "user", prompt, img_data)
            save_message(st.session_state.conversation_id, "assistant", res)
            end = time.time()
            log_step("POST_PROCESSING", start, end, "SUCCESS")
        except Exception as e:
            end = time.time()
            log_step("POST_PROCESSING", start, end, "FAILED", e)

        st.session_state.chat.append({"role":"user","content":prompt,"image":img_data})
        st.session_state.chat.append({"role":"assistant","content":res,"image":None})

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("image"):
                st.image(base64.b64decode(m["image"]), width="stretch")
