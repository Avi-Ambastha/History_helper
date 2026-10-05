import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="NCERT History Tutor", page_icon="📚")
st.title("📚 NCERT History Tutor (Class 10)")
st.caption("Ask me anything about your Class 10 History syllabus!")

# --- 2. LOAD BACKEND RESOURCES (CACHED) ---
# @st.cache_resource prevents the database and model from reloading every time you type a message
@st.cache_resource
def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./ncert_history_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"] 
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
    
    template = """You are a helpful and friendly Class 10 History tutor for NCERT students.
    First, try to use the provided textbook excerpts to answer the student's doubt. 
    If the specific answer is NOT in the context, use your general knowledge of  History to answer, but keep the language and scope appropriate for a 10th-standard student.
    Answer directly, use bullet points, and then provide the surrounding context. 
    End with a guiding Socratic question.

    Context from NCERT Textbook:
    {context}

    Student's Doubt: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = load_rag_chain()

# --- 3. MANAGE CHAT HISTORY ---
# This saves the conversation so it doesn't disappear when the page updates
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Class 10 History tutor. What topic are we studying today?"}
    ]

# Display all previous messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. HANDLE NEW MESSAGES ---
# st.chat_input creates the text box at the bottom of the screen
if prompt := st.chat_input("Ask a doubt (e.g., What was the Rowlatt Act?)..."):
    
    # 1. Display the student's question in the UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate the AI's response
    with st.chat_message("assistant"):
        # st.write_stream makes the AI type out the answer word-by-word like ChatGPT!
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response chunk by chunk from Ollama
        for chunk in rag_chain.stream(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
            
        # Finalize the output without the blinking cursor
        response_placeholder.markdown(full_response)
        
    # 3. Save the AI's answer to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
