import streamlit as st
import base64
from openai import OpenAI
import time
import logging
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

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



def retrieve_relevant_chunks(query, documents, top_k=10):
    query_words = set(query.lower().split())

    scored = []
    for doc in documents:
        text = doc.page_content.lower()

        score = 0

   
        for word in query_words:
            if word in text:
                score += 2

      
        for word in query_words:
            for token in text.split():
                if word in token or token in word:
                    score += 1

        
        if any(char.isdigit() for char in text):
            score += 2

       
        if any(k in text for k in ["reward", "score", "goal", "pass", "interception"]):
            score += 2

       
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for score, doc in scored][:top_k]



def expand_query(prompt):
    base_query = prompt

    base_query += " reward design scoring goal pass interception losing possession opponent scoring values table points numbers"

    return base_query



def smart_rag_response(prompt, rag_context):

  
    if rag_context and len(rag_context.strip()) > 50:

        rag_prompt = f"""
You are an intelligent assistant.

IMPORTANT RULES:
- Use the provided context FIRST
- If numerical values or points are present → use them EXACTLY
- DO NOT assume or create new values
- Only use general knowledge if context is missing information

Context:
{rag_context}

Question:
{prompt}
"""
        return chat_llm(rag_prompt)


    else:
        fallback_prompt = f"""
Answer the following question using your general knowledge:

{prompt}
"""
        return chat_llm(fallback_prompt)


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

if "documents" not in st.session_state:
    st.session_state.documents = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


def optimize_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((512, 512))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def process_document(file):
    file_bytes = file.read()
    file_name = file.name.lower()

    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = create_conversation(
            st.session_state.user_id,
            "Document Chat"
        )

    save_document(st.session_state.conversation_id, file_bytes)

    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        if file_name.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(temp_path)

        elif file_name.endswith(".ppt") or file_name.endswith(".pptx"):
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(temp_path, strategy="fast")

        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(temp_path, encoding="utf-8")

        start = time.time()
        docs = loader.load()
        end = time.time()
        log_step("DOC_LOADING", start, end, "SUCCESS")

    except Exception as e:
        log_step("DOC_LOADING", start, time.time(), "FAILED", e)
        st.error(f"Doc error: {str(e)}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    st.session_state.documents = chunks

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
        st.session_state.documents = None
        st.session_state.uploaded_image = None
        st.rerun()

    for cid, title in get_conversations(st.session_state.user_id):
        if st.sidebar.button(title, key=f"chat_{cid}"):
            st.session_state.conversation_id = cid
            st.session_state.chat = get_messages(cid)
            st.rerun()

    img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if img:
        st.session_state.uploaded_image = optimize_image(img.read())
        st.session_state.documents = None
        st.success("Image ready")

    doc = st.file_uploader("Upload Doc", type=["pdf","txt","ppt","pptx"])

    if doc:
        st.session_state.uploaded_image = None
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

        if st.session_state.documents:

            better_query = expand_query(prompt)

            docs = retrieve_relevant_chunks(
                better_query,
                st.session_state.documents,
                top_k=10
            )

            if docs:
                rag_context = "\n".join([d.page_content for d in docs])
            else:
                rag_context = ""

            print("\n===== RETRIEVED CONTEXT =====\n")
            print(rag_context)
            print("\n=============================\n")

        start = time.time()


        if st.session_state.uploaded_image:

            img_base64 = base64.b64encode(st.session_state.uploaded_image).decode()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt[:3000]},
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

        elif st.session_state.documents:
            res = smart_rag_response(prompt, rag_context)


        else:
            res = chat_llm(prompt)

        log_step("LLM_GENERATION", start, time.time(), "SUCCESS")

        save_message(st.session_state.conversation_id, "user", prompt)
        save_message(st.session_state.conversation_id, "assistant", res)

        st.session_state.chat.append({"role":"user","content":prompt})
        st.session_state.chat.append({"role":"assistant","content":res})

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
