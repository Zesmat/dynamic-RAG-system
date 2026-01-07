import os
import time
import json
import uuid
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.stores import InMemoryStore

try:
    from langchain_community.document_compressors import FlashrankRerank
    FlashrankRerank.model_rebuild()
except ImportError:
    pass

load_dotenv()

# --- Session Store with Thread ID Support ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create the chat history container for a specific thread."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class ThreadData:
    """Container for thread-specific data including PDFs, vector store, and chain."""
    def __init__(self, thread_id: str, thread_name: str):
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.created_at = datetime.now().isoformat()
        self.processed_files = []
        self.vector_store = None
        self.chain = None
        self.splits = None
        self.parent_store = InMemoryStore()
        self.persist_dir = f"./chroma_db_threads/{thread_id}"
        self.stats = {
            "total_chunks": 0,
            "total_pages": 0,
            "processing_time": 0,
            "queries_count": 0
        }
        # Create isolated directory for this thread
        os.makedirs(self.persist_dir, exist_ok=True)


class AdvancedRAGSystem:
    def __init__(self):
        """Initialize RAG system with multi-thread support and isolated PDF collections."""
        self.llm = None
        self.embeddings = None
        # Dictionary to store ThreadData objects
        self.threads: Dict[str, ThreadData] = {}
        self.global_stats = {
            "total_threads": 0,
            "total_pdfs_processed": 0
        }

    def create_new_thread(self, thread_name: Optional[str] = None) -> str:
        """Create a new conversation thread with its own isolated PDF collection."""
        thread_id = str(uuid.uuid4())
        name = thread_name or f"Thread {len(self.threads) + 1}"
        
        # Create new thread data container
        self.threads[thread_id] = ThreadData(thread_id, name)
        
        # Initialize the chat history for this thread
        get_session_history(thread_id)
        
        self.global_stats["total_threads"] += 1
        
        return thread_id

    def get_thread_info(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific thread."""
        if thread_id not in self.threads:
            return None
        
        thread = self.threads[thread_id]
        history = get_session_history(thread_id)
        
        return {
            "name": thread.thread_name,
            "created_at": thread.created_at,
            "message_count": len(history.messages),
            "pdf_count": len(thread.processed_files),
            "pdfs": thread.processed_files,
            "total_chunks": thread.stats["total_chunks"],
            "total_pages": thread.stats["total_pages"]
        }

    def list_threads(self) -> List[Dict[str, Any]]:
        """List all threads with their metadata."""
        threads = []
        for thread_id in self.threads:
            info = self.get_thread_info(thread_id)
            if info:
                threads.append({"id": thread_id, **info})
        return threads

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its associated data."""
        if thread_id not in self.threads:
            return False
        
        thread = self.threads[thread_id]
        
        # Delete the vector store directory
        if os.path.exists(thread.persist_dir):
            shutil.rmtree(thread.persist_dir)
        
        # Delete chat history
        if thread_id in store:
            del store[thread_id]
        
        # Delete thread data
        del self.threads[thread_id]
        
        return True

    def _extract_metadata(self, text: str, page_num: int) -> Dict[str, Any]:
        """Derive lightweight structural metadata for a text chunk."""
        metadata = {"page": page_num}
        
        text_lower = text.lower()
        if any(word in text_lower[:100] for word in ["abstract", "summary"]):
            metadata["section"] = "abstract"
        elif any(word in text_lower[:100] for word in ["introduction", "background"]):
            metadata["section"] = "introduction"
        elif any(word in text_lower[:100] for word in ["method", "methodology", "approach"]):
            metadata["section"] = "methods"
        elif any(word in text_lower[:100] for word in ["result", "finding", "outcome"]):
            metadata["section"] = "results"
        elif any(word in text_lower[:100] for word in ["conclusion", "discussion"]):
            metadata["section"] = "conclusion"
        elif any(word in text_lower[:100] for word in ["chapter", "section"]):
             metadata["section"] = "chapter"
        elif "prologue" in text_lower[:100] or "preface" in text_lower[:100]:
             metadata["section"] = "front_matter"
        elif "epilogue" in text_lower[:100] or "afterword" in text_lower[:100]:
             metadata["section"] = "back_matter"
        elif "glossary" in text_lower[:100] or "index" in text_lower[:100]:
             metadata["section"] = "reference"
        elif "appendix" in text_lower[:100]:
             metadata["section"] = "appendix"
        else:
            metadata["section"] = "general"
        
        words = text.split()
        metadata["chunk_size"] = len(words)
        
        return metadata

    def _create_hierarchical_chunks(self, docs: List[Document], thread: ThreadData) -> tuple:
        """Split documents into parent and child chunks for a specific thread."""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        parent_docs = parent_splitter.split_documents(docs)
        child_docs = []
        
        for parent_idx, parent_doc in enumerate(parent_docs):
            parent_id = f"parent_{parent_idx}_{uuid.uuid4().hex[:8]}"
            children = child_splitter.split_documents([parent_doc])
            
            for child in children:
                child.metadata.update(parent_doc.metadata)
                child.metadata["parent_id"] = parent_id
                child.metadata.update(
                    self._extract_metadata(child.page_content, child.metadata.get("page", 0))
                )
            
            child_docs.extend(children)
            thread.parent_store.mset([(parent_id, parent_doc)])
        
        return parent_docs, child_docs

    def process_pdf(self, pdf_file, thread_id: str, use_hierarchical: bool = True, 
                    use_metadata: bool = True):
        """Add a PDF to a specific thread's collection."""
        if not pdf_file:
            return "⚠️ Please upload a file.", self._get_thread_stats_display(thread_id)

        if thread_id not in self.threads:
            return "⚠️ Invalid thread ID.", self._get_thread_stats_display(thread_id)

        thread = self.threads[thread_id]
        start_time = time.time()
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_file.name)
            docs = loader.load()
            
            file_name = os.path.basename(pdf_file.name)
            
            # Check if file already processed in this thread
            if file_name in thread.processed_files:
                return f"⚠️ {file_name} already processed in this thread.", self._get_thread_stats_display(thread_id)
            
            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["thread_id"] = thread_id
            
            pages_in_pdf = len(docs)
            
            # Create chunks
            if use_hierarchical:
                parent_docs, child_docs = self._create_hierarchical_chunks(docs, thread)
                new_splits = child_docs
                status_prefix = "🔗 Hierarchical"
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=120,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                new_splits = text_splitter.split_documents(docs)
                
                if use_metadata:
                    for split in new_splits:
                        split.metadata.update(
                            self._extract_metadata(split.page_content, split.metadata.get("page", 0))
                        )
                status_prefix = "📄 Standard"
            
            # Initialize embeddings if not already done
            if not self.embeddings:
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Add to existing vector store or create new one
            if thread.vector_store is None:
                # First PDF in this thread - create new vector store
                thread.vector_store = Chroma.from_documents(
                    documents=new_splits,
                    embedding=self.embeddings,
                    persist_directory=thread.persist_dir
                )
                thread.splits = new_splits
            else:
                # Add to existing vector store
                thread.vector_store.add_documents(new_splits)
                thread.splits.extend(new_splits)
            
            # Rebuild chain with updated documents
            self._build_chain(thread_id)
            
            # Update thread stats
            thread.processed_files.append(file_name)
            thread.stats["total_chunks"] = len(thread.splits)
            thread.stats["total_pages"] += pages_in_pdf
            thread.stats["processing_time"] = round(time.time() - start_time, 2)
            
            self.global_stats["total_pdfs_processed"] += 1
            
            return (
                f"✅ {status_prefix} | Added {file_name}: {len(new_splits)} chunks from {pages_in_pdf} pages\n"
                f"📚 Thread now has {len(thread.processed_files)} PDF(s) with {thread.stats['total_chunks']} total chunks",
                self._get_thread_stats_display(thread_id)
            )
            
        except Exception as e:
            return f"❌ Error: {str(e)}", self._get_thread_stats_display(thread_id)

    def _build_chain(self, thread_id: str):
        """Build the RAG chain for a specific thread."""
        if thread_id not in self.threads:
            return
        
        thread = self.threads[thread_id]
        
        if not self.llm:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                temperature=0.3
            )

        vector_retriever = thread.vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 8, "fetch_k": 40}
        )

        bm25_retriever = BM25Retriever.from_documents(thread.splits)
        bm25_retriever.k = 10

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )

        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            (MessagesPlaceholder("chat_history")),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, compression_retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are a precise assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise but comprehensive.
        
        Context:
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            (MessagesPlaceholder("chat_history")),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        thread.chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate alternative phrasings of a query."""
        if not self.llm:
            return [query]
        
        prompt = f"""Generate 2 alternative phrasings of this question that capture the same intent:

                    Original: {query}

                    Alternative 1:
                    Alternative 2:

                    Only output the alternatives, one per line."""
        
        try:
            response = self.llm.invoke(prompt)
            variations = [query] + [line.strip() for line in response.content.split('\n') if line.strip()]
            return variations[:3]
        except:
            return [query]

    def chat(self, message, thread_id: str, use_query_expansion: bool = False, 
             section_filter: Optional[str] = None, page_filter: Optional[str] = None):
        """Process a chat message within a specific thread."""
        if thread_id not in self.threads:
            return "⚠️ Invalid thread ID."
        
        thread = self.threads[thread_id]
        
        if not thread.chain:
            return "⚠️ Please upload and process a PDF first for this thread."

        config = {"configurable": {"session_id": thread_id}}
        max_retries = 3
        attempt = 0
        
        thread.stats["queries_count"] += 1

        while attempt < max_retries:
            try:
                if use_query_expansion:
                    query_variations = self._generate_query_variations(message)
                    message_to_use = " OR ".join(query_variations)
                else:
                    message_to_use = message
                
                response = thread.chain.invoke({"input": message_to_use}, config=config)
                answer = response["answer"]
                
                sources_detail = []
                seen_sources = set()
                
                for doc in response["context"]:
                    page = doc.metadata.get('page', '?')
                    section = doc.metadata.get('section', 'general')
                    source = doc.metadata.get('source', 'Unknown')
                    
                    if section_filter and section_filter != "all" and section != section_filter:
                        continue
                    if page_filter and page_filter.strip():
                        try:
                            filter_pages = [int(p.strip()) for p in page_filter.split(',')]
                            if page not in filter_pages:
                                continue
                        except:
                            pass
                    
                    source_key = f"{source}_p{page}_{section}"
                    if source_key not in seen_sources:
                        sources_detail.append(f"📄 {source} | Page {page} | {section.title()}")
                        seen_sources.add(source_key)
                
                source_str = "\n\n**Sources:**\n" + "\n".join(sources_detail[:5]) if sources_detail else ""
                
                confidence = min(95, 60 + len(response["context"]) * 4)
                confidence_emoji = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
                confidence_str = f"\n\n{confidence_emoji} Confidence: ~{confidence}%"

                return answer + source_str + confidence_str
            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    attempt += 1
                    if attempt < max_retries:
                        wait_time = 60
                        print(f"⚠️ Quota exceeded. Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "❌ Error: API Quota exceeded even after retries. Please try again later."
                else:
                    return f"❌ Error: {error_msg}"

    def _get_thread_stats_display(self, thread_id: str) -> str:
        """Render statistics for a specific thread."""
        if thread_id not in self.threads:
            return "⚠️ Invalid thread"
        
        thread = self.threads[thread_id]
        
        return f"""
📊 **Thread Statistics: {thread.thread_name}**
- 📚 PDFs in Thread: {len(thread.processed_files)}
- 📄 Total Pages: {thread.stats['total_pages']}
- 🧩 Total Chunks: {thread.stats['total_chunks']}
- ⏱️ Last Processing: {thread.stats['processing_time']}s
- 💬 Queries in Thread: {thread.stats['queries_count']}

📄 **PDFs:**
{chr(10).join(f"  • {pdf}" for pdf in thread.processed_files) if thread.processed_files else "  None yet"}
"""

    def _get_global_stats_display(self) -> str:
        """Render global system statistics."""
        return f"""
🌐 **Global Statistics**
- 🧵 Total Threads: {len(self.threads)}
- 📚 Total PDFs Processed: {self.global_stats['total_pdfs_processed']}
"""

    def export_conversation(self, thread_id: str) -> str:
        """Export a specific thread's conversation to JSON."""
        if thread_id not in self.threads:
            return None
        
        thread = self.threads[thread_id]
        history = get_session_history(thread_id)
        messages = []
        
        for msg in history.messages:
            messages.append({
                "role": msg.type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "thread_id": thread_id,
            "thread_name": thread.thread_name,
            "created_at": thread.created_at,
            "documents": thread.processed_files,
            "stats": thread.stats,
            "conversation": messages
        }
        
        filename = f"rag_thread_{thread_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

    def export_all_threads(self) -> str:
        """Export all threads to a single JSON file."""
        all_threads = {}
        
        for thread_id, thread in self.threads.items():
            history = get_session_history(thread_id)
            messages = []
            
            for msg in history.messages:
                messages.append({
                    "role": msg.type,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })
            
            all_threads[thread_id] = {
                "name": thread.thread_name,
                "created_at": thread.created_at,
                "documents": thread.processed_files,
                "stats": thread.stats,
                "conversation": messages
            }
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_threads": len(all_threads),
            "global_stats": self.global_stats,
            "threads": all_threads
        }
        
        filename = f"rag_all_threads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename


# --- Gradio UI with Thread Management ---
rag_system = AdvancedRAGSystem()

# Create default thread
default_thread_id = rag_system.create_new_thread("Default Conversation")

with gr.Blocks(title="Advanced RAG System") as demo:
    gr.Markdown("# 🧠 Advanced Hybrid RAG System with Isolated Thread Collections")
    gr.Markdown("*Each thread maintains its own isolated PDF collection - no cross-contamination!*")
    
    # Store current thread ID in state
    current_thread = gr.State(value=default_thread_id)
    
    with gr.Tabs():
        with gr.Tab("📤 Document Upload"):
            gr.Markdown("### Upload PDFs to Current Thread")
            gr.Markdown("Each PDF you upload will be added to the current thread's isolated collection.")
            
            current_thread_name = gr.Textbox(
                label="Current Thread",
                value=rag_system.threads[default_thread_id].thread_name,
                interactive=False
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                    
                    with gr.Row():
                        use_hierarchical = gr.Checkbox(label="🔗 Hierarchical Chunking", value=True)
                        use_metadata = gr.Checkbox(label="🏷️ Metadata Extraction", value=True)
                    
                    upload_btn = gr.Button("🚀 Process Document", variant="primary")
                
                with gr.Column(scale=1):
                    stats_display = gr.Markdown(rag_system._get_thread_stats_display(default_thread_id))
            
            status_msg = gr.Textbox(
                label="Processing Status", 
                value="⏳ Waiting for document upload...", 
                interactive=False
            )
        
        with gr.Tab("💬 Chat Interface"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Thread management
                    with gr.Row():
                        thread_name_input = gr.Textbox(
                            label="New Thread Name (optional)",
                            placeholder="My Research Questions",
                            scale=2
                        )
                        new_thread_btn = gr.Button("➕ Create New Thread", scale=1)
                    
                    current_thread_display = gr.Textbox(
                        label="Current Thread ID",
                        value=default_thread_id,
                        interactive=False
                    )
                    
                    thread_info = gr.Markdown(rag_system._get_thread_stats_display(default_thread_id))
                    
                    gr.Markdown("### Ask questions about PDFs uploaded to this thread")
                    
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=400
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about your documents...",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    clear_btn = gr.Button("🗑️ Clear Chat Display")
                    
                    gr.Markdown("#### Example Questions")
                    with gr.Row():
                        example1 = gr.Button("What are the main findings?", size="sm")
                        example2 = gr.Button("Summarize the methodology", size="sm")
                    with gr.Row():
                        example3 = gr.Button("What are the key conclusions?", size="sm")
                        example4 = gr.Button("Compare information across documents", size="sm")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Query Settings")
                    query_expansion = gr.Checkbox(label="🔄 Query Expansion", value=False)
                    
                    gr.Markdown("### 🔍 Filters")
                    section_filter = gr.Dropdown(
                        choices=["all", "abstract", "introduction", "methods", "results", "conclusion"],
                        value="all",
                        label="Section Filter"
                    )
                    page_filter = gr.Textbox(
                        label="Page Filter (e.g., 1,3,5)",
                        placeholder="Leave empty for all pages"
                    )
                    
                    gr.Markdown("### 💾 Export Options")
                    export_current_btn = gr.Button("Export Current Thread")
                    export_all_btn = gr.Button("Export All Threads")
                    export_output = gr.Textbox(label="Export Status", interactive=False)
        
        with gr.Tab("🧵 Thread Management"):
            gr.Markdown("### Manage Conversation Threads")
            gr.Markdown("**Key Feature:** Each thread has its own isolated PDF collection!")
            
            global_stats = gr.Markdown(rag_system._get_global_stats_display())
            
            refresh_threads_btn = gr.Button("🔄 Refresh Thread List")
            threads_display = gr.JSON(label="Active Threads", value=rag_system.list_threads())
            
            with gr.Row():
                thread_selector = gr.Dropdown(
                    label="Switch to Thread",
                    choices=[(t["name"], t["id"]) for t in rag_system.list_threads()],
                    interactive=True
                )
                switch_thread_btn = gr.Button("Switch Thread")
            
            with gr.Row():
                delete_thread_selector = gr.Dropdown(
                    label="Delete Thread",
                    choices=[(t["name"], t["id"]) for t in rag_system.list_threads() if t["id"] != default_thread_id],
                    interactive=True
                )
                delete_thread_btn = gr.Button("🗑️ Delete Thread", variant="stop")
            
            switch_status = gr.Textbox(label="Status", interactive=False)

    # Event handlers
    def upload_pdf_to_thread(file, thread_id, hierarchical, metadata):
        result, stats = rag_system.process_pdf(file, thread_id, hierarchical, metadata)
        thread_name = rag_system.threads[thread_id].thread_name if thread_id in rag_system.threads else "Unknown"
        return result, stats, thread_name
    
    def chat_fn(message, history, thread_id):
        """Handle chat messages and update history."""
        # Convert history from Gradio format (list of dicts) to simple tuples for RAG system
        # History is a list like [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, ...]
        simple_history = []
        if history:
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    user_msg = history[i]["content"] if isinstance(history[i], dict) else history[i]
                    assistant_msg = history[i+1]["content"] if isinstance(history[i+1], dict) else history[i+1]
                    simple_history.append((user_msg, assistant_msg))
        
        response = rag_system.chat(
            message, thread_id,
            use_query_expansion=query_expansion.value,
            section_filter=section_filter.value,
            page_filter=page_filter.value
        )
        
        # Append in Gradio's expected format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""
    
    def clear_chat():
        """Clear the chat display."""
        return []
    
    upload_btn.click(
        fn=upload_pdf_to_thread,
        inputs=[file_input, current_thread, use_hierarchical, use_metadata],
        outputs=[status_msg, stats_display, current_thread_name]
    )
    
    def create_thread(name):
        thread_id = rag_system.create_new_thread(name if name else None)
        thread_name = rag_system.threads[thread_id].thread_name
        threads = rag_system.list_threads()
        choices = [(t["name"], t["id"]) for t in threads]
        delete_choices = [(t["name"], t["id"]) for t in threads if t["id"] != default_thread_id]
        stats = rag_system._get_thread_stats_display(thread_id)
        global_stats = rag_system._get_global_stats_display()
        return (thread_id, thread_id, thread_name, stats, 
                gr.update(choices=choices), threads, global_stats,
                gr.update(choices=delete_choices))
    
    new_thread_btn.click(
        fn=create_thread,
        inputs=[thread_name_input],
        outputs=[current_thread, current_thread_display, current_thread_name, 
                thread_info, thread_selector, threads_display, global_stats,
                delete_thread_selector]
    )
    
    def refresh_threads():
        threads = rag_system.list_threads()
        choices = [(t["name"], t["id"]) for t in threads]
        delete_choices = [(t["name"], t["id"]) for t in threads if t["id"] != default_thread_id]
        global_stats = rag_system._get_global_stats_display()
        return threads, gr.update(choices=choices), global_stats, gr.update(choices=delete_choices)
    
    def switch_thread(selected_thread_id):
        if selected_thread_id and selected_thread_id in rag_system.threads:
            thread_name = rag_system.threads[selected_thread_id].thread_name
            stats = rag_system._get_thread_stats_display(selected_thread_id)
            return (selected_thread_id, selected_thread_id, thread_name, stats,
                    f"✅ Switched to thread: {thread_name}", [])
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                "⚠️ Please select a valid thread", gr.update())
    
    # Chat event handlers
    submit_btn.click(
        fn=chat_fn,
        inputs=[msg_input, chatbot, current_thread],
        outputs=[chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=chat_fn,
        inputs=[msg_input, chatbot, current_thread],
        outputs=[chatbot, msg_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )
    
    # Example button handlers
    example1.click(lambda: "What are the main findings?", outputs=msg_input)
    example2.click(lambda: "Summarize the methodology", outputs=msg_input)
    example3.click(lambda: "What are the key conclusions?", outputs=msg_input)
    example4.click(lambda: "Compare information across documents", outputs=msg_input)
    
    refresh_threads_btn.click(
        fn=refresh_threads,
        outputs=[threads_display, thread_selector, global_stats, delete_thread_selector]
    )
    
    switch_thread_btn.click(
        fn=switch_thread,
        inputs=[thread_selector],
        outputs=[current_thread, current_thread_display, current_thread_name, 
                thread_info, switch_status, chatbot]
    )
    
    def delete_thread_handler(thread_id_to_delete, current_tid):
        if not thread_id_to_delete:
            return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Please select a thread to delete"
        
        if thread_id_to_delete == default_thread_id:
            return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete default thread"
        
        if thread_id_to_delete == current_tid:
            return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete current active thread. Switch to another thread first."
        
        if rag_system.delete_thread(thread_id_to_delete):
            threads = rag_system.list_threads()
            choices = [(t["name"], t["id"]) for t in threads]
            delete_choices = [(t["name"], t["id"]) for t in threads if t["id"] != default_thread_id]
            global_stats = rag_system._get_global_stats_display()
            return (threads, gr.update(choices=choices), global_stats,
                    gr.update(choices=delete_choices),
                    f"✅ Thread deleted successfully")
        return gr.update(), gr.update(), gr.update(), gr.update(), "❌ Failed to delete thread"
    
    delete_thread_btn.click(
        fn=delete_thread_handler,
        inputs=[delete_thread_selector, current_thread],
        outputs=[threads_display, thread_selector, global_stats, 
                delete_thread_selector, switch_status]
    )
    
    export_current_btn.click(
        fn=lambda tid: f"✅ Exported to: {rag_system.export_conversation(tid)}" if tid in rag_system.threads else "❌ Invalid thread",
        inputs=[current_thread],
        outputs=export_output
    )
    
    export_all_btn.click(
        fn=lambda: f"✅ Exported all threads to: {rag_system.export_all_threads()}",
        outputs=export_output
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())