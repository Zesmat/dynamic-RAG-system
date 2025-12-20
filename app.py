
# LangChain Imports
import os
import gradio as gr
from dotenv import load_dotenv
import time
# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# LangChain Imports

from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
# 1. Load the Environment Variables (API Key)
load_dotenv()

# Global variable to hold the database temporarily
vector_store = None

def process_pdf(pdf_file):
    """
    Reads the PDF, splits it into chunks, and creates the vector database.
    """
    global vector_store
    
    if pdf_file is None:
        return "Please upload a file first."

    try:
        # A. Load the PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        # B. Split the text (Chunking Strategy: Recursive)
        # We start with 1000 characters per chunk, with 200 overlap to keep context.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents)

        # C. Create Embeddings & Database
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # This creates the database in memory (RAM) so it's fast and temporary
        vector_store = Chroma.from_documents(chunks, embeddings)

        return f"Success! Processed {len(chunks)} chunks. You can now ask questions."
    
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(question):
    """
    Searches the database for relevant info and asks Gemini to answer.
    """
    global vector_store

    if vector_store is None:
        return "Please upload and process a PDF document first."
    
    if not question:
        return "Please type a question."
    
    
    time.sleep(4)

    # A. Setup the LLM (Gemini Flash for speed)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", temperature=0.3)

    # B. Create the Chain (The RAG Logic)
    # This chain retrieves data from vector_store and sends it to the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' means put all found text into one prompt
        retriever=vector_store.as_retriever()
    )

    # C. Get the Answer
    response = qa_chain.run(question)
    return response

# --- The User Interface (Gradio) ---
with gr.Blocks(title="Dynamic RAG Analyzer") as demo:
    gr.Markdown("# 🤖 Dynamic Textbook Analyzer")
    gr.Markdown("Upload a PDF, process it, and ask questions about it.")
    
    with gr.Row():
        # Column 1: Upload
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        # Column 2: Chat
        with gr.Column():
            question_input = gr.Textbox(label="Ask a Question")
            ask_btn = gr.Button("Ask AI")
            answer_output = gr.Textbox(label="Answer")

    # Connect the buttons to functions
    process_btn.click(fn=process_pdf, inputs=pdf_input, outputs=status_output)
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

# Launch the app
if __name__ == "__main__":
    demo.launch()