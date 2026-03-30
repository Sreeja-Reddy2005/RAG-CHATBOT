import streamlit as st
import base64
from openai import OpenAI

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from auth import login, register
from chat_db import *

from PIL import Image
import io


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="MY_TOKEN"
)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic"
FALLBACK = "meta-llama/Llama-3.1-8B-Instruct"

create_tables()


if "chat" not in st.session_state:
    st.session_state.chat = []

if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()

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

        docs = loader.load()

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
            max_tokens=500
        )
        return res.choices[0].message.content
    except:
        res = client.chat.completions.create(
            model=FALLBACK,
            messages=[{"role": "user", "content": prompt[:3000]}],
            max_tokens=500
        )
        return res.choices[0].message.content


def generate_full_summary():

    if not st.session_state.chat:
        return "No conversation found."

    ordered_points = []

    for i, m in enumerate(st.session_state.chat, 1):
        role = m["role"]
        content = m["content"]


        point = f"Step {i}: {role} discussed - {content[:80]}"
        ordered_points.append(point)

        if m.get("image"):
            ordered_points.append(f"Step {i}: Image was analyzed")

    steps_text = "\n".join(ordered_points)

    summary = chat_llm(f"""
You are given a conversation in STRICT ORDER.

These are the steps (DO NOT SKIP ANY):
{steps_text}

YOUR TASK:
Write ONE paragraph summary that follows EXACT SAME ORDER.

STRICT RULES:
- Follow Step 1 → Step N in order
- EVERY step MUST appear in summary
- DO NOT skip final steps (IMPORTANT)
- DO NOT prioritize topics
- DO NOT merge unrelated steps
- Include ALL topics (including Angular at end)
- Include image analysis
- Keep natural paragraph flow

If ANY step is missing → answer is WRONG.

Final Paragraph Summary:
""")

    return summary

st.title("Multimodal RAG Chatbot")

if "user_id" not in st.session_state:
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Register"):
        if register(u, p):
            st.success("Registered")

    if st.button("Login"):
        user = login(u, p)
        if user:
            st.session_state.user_id = user.id
            st.rerun()

else:
    st.sidebar.title("Chats")

    # RESET CHAT
    if st.sidebar.button("New Chat"):
        st.session_state.chat = []
        st.session_state.memory = ChatMessageHistory()
        st.session_state.conversation_id = None
        st.session_state.vectorstore = None
        st.session_state.uploaded_image = None
        st.session_state.use_image_next = False
        st.rerun()

    # SUMMARY BUTTON
    if st.sidebar.button("Full Session Summary"):
        summary = generate_full_summary()

        st.session_state.chat.append({
            "role": "assistant",
            "content": f" Summary:\n\n{summary}",
            "image": None
        })

        if st.session_state.conversation_id:
            save_summary(st.session_state.conversation_id, summary)

        st.rerun()

    # LOAD HISTORY
    for cid, title in get_conversations(st.session_state.user_id):
        if st.sidebar.button(title, key=f"chat_{cid}"):

            st.session_state.conversation_id = cid
            msgs = get_messages(cid)

            st.session_state.chat = msgs
            st.rerun()

    # IMAGE
    img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if img:
        try:
            st.session_state.uploaded_image = optimize_image(img.read())
            st.session_state.use_image_next = True
            st.success("Image ready")
        except:
            st.error("Invalid image")

    # DOC
    doc = st.file_uploader("Upload Doc", type=["pdf","txt"])

    if doc:
        process_document(doc)
    else:
        st.session_state.vectorstore = None

    prompt = st.chat_input("Ask...")

    if prompt:

        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(
                st.session_state.user_id,
                prompt[:30]
            )

        img_data = None
        if st.session_state.use_image_next and st.session_state.uploaded_image:
            img_data = base64.b64encode(st.session_state.uploaded_image).decode()


        rag_context = ""
        use_doc = False

        if st.session_state.vectorstore is not None:

            if "analyze" in prompt.lower() or "analyse" in prompt.lower():
                docs = st.session_state.vectorstore.similarity_search(
                    "summary of document", k=3
                )
                use_doc = True
            else:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=2)

                if docs:
                    use_doc = True

            rag_context = "\n".join([d.page_content for d in docs])

 
        if use_doc and rag_context.strip():
            final_prompt = f"""
Use ONLY this document context:

{rag_context}

Question:
{prompt}

Answer clearly:
"""
        else:
            final_prompt = f"""
Answer normally:

Question:
{prompt}
"""


        if st.session_state.use_image_next and st.session_state.uploaded_image:

            img_base64 = base64.b64encode(st.session_state.uploaded_image).decode()

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt[:3000]},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }],
                    max_tokens=500
                )

                res = response.choices[0].message.content

            except:
                res = chat_llm(final_prompt)


            st.session_state.use_image_next = False
            st.session_state.uploaded_image = None

        else:
            res = chat_llm(final_prompt)


        save_message(st.session_state.conversation_id, "user", prompt, img_data)
        save_message(st.session_state.conversation_id, "assistant", res)

        st.session_state.chat.append({"role":"user","content":prompt,"image":img_data})
        st.session_state.chat.append({"role":"assistant","content":res,"image":None})


    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("image"):
                st.image(base64.b64decode(m["image"]))