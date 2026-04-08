import streamlit as st
import base64
from openai import OpenAI
from auth import login, register
from chat_db import *

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from PIL import Image
import io
import base64
import os


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="my_token"
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"

def chat_llm(prompt):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt[:4000]}],
    )
    return res.choices[0].message.content



def process_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()



def retrieve_relevant_chunks(query, documents, top_k=10):
    query_words = query.lower().split()
    scored = []

    for doc in documents:
        text = doc.page_content.lower()
        score = 0

        for word in query_words:
            if word in text:
                score += 2

        for word in query_words:
            if any(word in token for token in text.split()):
                score += 1

        score += sum(1 for word in query_words if word in text)

        if any(char.isdigit() for char in text):
            score += 3

        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def rerank_chunks(query, retrieved_chunks, top_k=3):
    return [doc for _, doc in retrieved_chunks[:top_k]]


def expand_query(prompt):
    return prompt + " reward score goal pass interception table values"


def smart_rag_response(prompt, rag_context, is_general_query):

    if not rag_context:
        return chat_llm(prompt)

    if is_general_query:
        rag_prompt = f"""
Give COMPLETE analysis of the document.

Context:
{rag_context}
"""
    else:
        rag_prompt = f"""
Answer ONLY using the context.

Context:
{rag_context}

Question:
{prompt}
"""

    return chat_llm(rag_prompt)



create_tables()

if "chat" not in st.session_state:
    st.session_state.chat = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "image" not in st.session_state:
    st.session_state.image = None



if "user_id" not in st.session_state:

    st.title("Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login(u,p)
        if user:
            st.session_state.user_id = user.id
            st.rerun()

    if st.button("Register"):
        register(u,p)
        st.success("Registered")


else:

    st.sidebar.title("Chats")

    if st.sidebar.button("New Chat"):
        st.session_state.chat = []
        st.session_state.conversation_id = None
        st.session_state.documents = None
        st.session_state.image = None
        st.rerun()

  
    file_path = None
    if st.session_state.conversation_id:
        file_path = load_document_path(st.session_state.conversation_id)
        if file_path:
            st.sidebar.write("📄", file_path.split("/")[-1])

    for cid, title in get_conversations(st.session_state.user_id):
        if st.sidebar.button(title, key=cid):
            st.session_state.conversation_id = cid
            st.session_state.chat = get_messages(cid)

            file_path = load_document_path(cid)

            if file_path:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
                st.session_state.documents = splitter.split_documents(docs)
            else:
                st.session_state.documents = None

            img_base64 = load_image(cid)

            if img_base64:
                st.session_state.image = base64.b64decode(img_base64)
            else:
                st.session_state.image = None

            st.rerun()

    img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if img:
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(
                st.session_state.user_id, "Image Chat"
            )

        img_bytes = process_image(img.read())
        img_base64 = base64.b64encode(img_bytes).decode()

        save_image(st.session_state.conversation_id, img_base64)

        st.session_state.image = img_bytes
        st.session_state.documents = None

        st.success("Image ready!")


    doc = st.file_uploader("Upload PDF")

    if doc:
        
        file_bytes = doc.read()

        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(
                st.session_state.user_id, "Doc Chat"
            )
        

        os.makedirs("docs", exist_ok=True)

        file_path = f"docs/{doc.name}"

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        save_document(st.session_state.conversation_id, file_path)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        st.session_state.documents = splitter.split_documents(docs)

        st.session_state.image = None

        st.success("Document ready!")

   
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask...")

    if prompt:

        if len(prompt.split()) <= 3:
            prompt = f"Explain in detail: {prompt}"

        is_general_query = any(word in prompt.lower() for word in [
            "analyse", "analyze", "summary", "overview"
        ])

        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(
                st.session_state.user_id,
                prompt[:50]
            )

        if st.session_state.image:
            img_base64 = base64.b64encode(st.session_state.image).decode()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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
            query = expand_query(prompt)
            retrieved = retrieve_relevant_chunks(query, st.session_state.documents)

            if retrieved:
                best_docs = rerank_chunks(prompt, retrieved)
                rag_context = "\n".join([d.page_content for d in best_docs])
            else:
                rag_context = ""

            res = smart_rag_response(prompt, rag_context, is_general_query)

        else:
            res = chat_llm(prompt)

        save_message(st.session_state.conversation_id, "user", prompt)
        save_message(st.session_state.conversation_id, "assistant", res)

        st.session_state.chat.append({"role":"user","content":prompt})
        st.session_state.chat.append({"role":"assistant","content":res})

        st.rerun()
