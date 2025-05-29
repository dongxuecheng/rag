import os
import shutil # Import shutil for file operations
import gradio as gr # Import Gradio
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Import ChatOpenAI for OpenAI-compatible endpoint
from langchain_openai import ChatOpenAI
# Import a Gradio theme
import gradio.themes as gr_themes

# --- Configuration ---
DOCUMENTS_DIR = "./documents"  # Modify to your document directory
PERSIST_DIR = "./chroma_db"     # Vector database storage directory
EMBEDDING_MODEL_PATH = "/mnt/dxc/model/bge-m3" # CRITICAL: Retrieval quality HEAVILY depends on this model. If issues persist, experimenting with other embedding models (e.g., from sentence-transformers) is STRONGLY recommended.
EMBEDDING_DEVICE = "cuda:1" # Or 'cpu'
# VLLM Server details (using OpenAI compatible endpoint)
VLLM_BASE_URL = "http://localhost:8000/v1" # Default for `vllm serve`
VLLM_API_KEY = "dummy-key" # Required by ChatOpenAI, but VLLM server doesn't usually check it
VLLM_MODEL_NAME = "/mnt/dxc/model/qwen3-8b" # The model name used in `vllm serve`
CHUNK_SIZE = 512 # Adjusted for bge-m3, which can handle more context
CHUNK_OVERLAP = 100  # Adjusted overlap (approx 20% of CHUNK_SIZE)
SEARCH_K = 10 # Retrieve more chunks to increase chances of finding specific sentences
# --- End Configuration ---

# Global variables
rag_chain = None
vector_db = None
embeddings = None
llm = None

# 1. 定义文档加载函数，支持PDF和Word
def load_documents(directory_path):
    documents = []

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)

        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

    return documents

# 2. 文本分割
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",  # Split by double newlines (paragraphs)
            "\n",    # Split by single newlines
            ". ",    # Split by period followed by space (ensure space to avoid splitting mid-sentence e.g. Mr. Smith)
            "? ",    # Split by question mark followed by space
            "! ",    # Split by exclamation mark followed by space
            "。 ",   # Chinese period followed by space (if applicable)
            "？ ",   # Chinese question mark followed by space (if applicable)
            "！ ",   # Chinese exclamation mark followed by space (if applicable)
            "。\n",  # Chinese period followed by newline
            "？\n",  # Chinese question mark followed by newline
            "！\n",  # Chinese exclamation mark followed by newline
            " ",     # Split by space as a fallback
            ""       # Finally, split by character if no other separator is found
        ],
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# 3. 初始化HuggingFace嵌入模型
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )

# 4. 创建或加载向量数据库 (Modified)
def get_vector_db(chunks, embeddings, persist_directory):
    """Creates a new vector DB or loads an existing one."""
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector database from {persist_directory}...")
        try:
            # When loading, ChromaDB will check for dimension compatibility.
            # If EMBEDDING_MODEL_PATH changed leading to a dimension mismatch, this will fail.
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception as e:
            print(f"Error loading existing vector database: {e}.")
            print(f"This might be due to a change in the embedding model and a dimension mismatch.")
            print(f"If you changed EMBEDDING_MODEL_PATH, you MUST delete the old database directory: {persist_directory}")
            # If loading fails, proceed as if it doesn't exist, but only create if chunks are given later.
            return None # Indicate loading failed or DB doesn't exist in a usable state
    else:
        # Directory doesn't exist or is empty
        if chunks:
            print(f"Creating new vector database in {persist_directory}...")
            print(f"Creating Chroma DB with {len(chunks)} chunks...")
            try:
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                print("Vector database created and persisted.")
                return vector_db
            except Exception as e:
                print(f"Error creating new vector database: {e}")
                raise  # Re-raise the exception if creation fails
        else:
            # No chunks provided and DB doesn't exist/is empty - cannot create.
            print(f"Vector database directory {persist_directory} not found or empty, and no chunks provided to create a new one.")
            return None # Indicate DB doesn't exist and cannot be created yet

# 5. 初始化连接到VLLM服务器的ChatOpenAI客户端 (Replaces initialize_llm)
def initialize_openai_client():
    """Initializes ChatOpenAI client pointing to the VLLM server."""
    print(f"Initializing ChatOpenAI client for VLLM server at {VLLM_BASE_URL}...")
    return ChatOpenAI(
        openai_api_base=VLLM_BASE_URL,
        openai_api_key=VLLM_API_KEY,
        model_name=VLLM_MODEL_NAME
    )

# 6. 创建RAG检索链（使用新方法）
def create_rag_chain(vector_db, llm):
    # 从Langchain Hub获取检索QA聊天提示
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # 创建文档组合链
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)
    # 创建检索链
    # Using default similarity search. If this fails, and embedding model is good,
    # then advanced retrieval or query transformation might be needed.
    retriever = vector_db.as_retriever(search_kwargs={"k": SEARCH_K})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain

# 7. Function to process query using the RAG chain (Modified for Streaming)
def process_query(query):
    """Processes a user query using the RAG chain and streams the answer."""
    global rag_chain, vector_db # Add vector_db to globals accessed here for debugging
    if rag_chain is None:
        yield "错误：RAG 链未初始化。"
        return

    # --- For Debugging Retrieval ---
    # Uncomment the block below to see what documents are retrieved by the vector DB
    if vector_db:
        try:
            retrieved_docs = vector_db.similarity_search(query, k=SEARCH_K)
            print(f"\n--- Retrieved Documents for query: '{query}' ---")
            for i, doc in enumerate(retrieved_docs):
                # Attempt to get score if retriever supports it (Chroma's similarity_search_with_score)
                # For basic similarity_search, score might not be directly in metadata.
                # If using retriever.get_relevant_documents(), score might be present.
                score = doc.metadata.get('score', 'N/A') # Placeholder, actual score retrieval might differ
                if hasattr(doc, 'score'): # Check if score attribute exists directly
                    score = doc.score
                
                print(f"Doc {i+1} (Score: {score}):")
                print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
                print(f"Metadata: {doc.metadata}")
            print("--- End Retrieved Documents ---\n")
        except Exception as e:
            print(f"Error during debug similarity_search: {e}")
    else:
        print("Vector DB not initialized, skipping debug retrieval.")
    # --- End Debugging Retrieval ---

    try:
        print(f"开始处理流式查询: {query}")

        # Directly stream from the RAG chain runnable
        # The input format for create_retrieval_chain is typically {"input": query}
        # The output chunks often contain 'answer' and 'context' keys
        response_stream = rag_chain.stream({"input": query})

        full_answer = ""
        # Yield chunks as they arrive. Gradio Textbox updates incrementally.
        print("开始流式生成回答...")
        for chunk in response_stream:
            # Check if the 'answer' key exists in the chunk and append it
            answer_part = chunk.get("answer", "")
            if answer_part:
                full_answer += answer_part
                # Debugging output
                # print(f"Raw answer_part from LLM: '{answer_part}'")
                # print(f"Yielding to Gradio: '{full_answer}'")
                yield full_answer # Yield the progressively built answer

        if not full_answer:
             yield "抱歉，未能生成回答。" # Handle cases where stream completes without answer

        print(f"流式处理完成。最终回答: {full_answer}")

    except Exception as e:
        print(f"处理查询时发生错误: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        yield f"处理查询时发生错误: {e}"

# 8. Function to rebuild the index and RAG chain (Modified to add documents)
def rebuild_index_and_chain():
    """Loads documents, creates/updates vector DB by adding new content, and rebuilds the RAG chain."""
    global vector_db, rag_chain, embeddings, llm

    if embeddings is None or llm is None:
        return "错误：Embeddings 或 LLM 未初始化。"

    # Ensure documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"创建文档目录: {DOCUMENTS_DIR}")

    # Step 1: Load documents
    print("加载文档...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"在 {DOCUMENTS_DIR} 中未找到文档。")
        # Try to load existing DB even if no new documents are found
        print("尝试加载现有向量数据库...")
        # Pass None for chunks as we are just trying to load
        vector_db = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db:
            print("没有新文档加载，将使用现有的向量数据库。重新创建 RAG 链...")
            rag_chain = create_rag_chain(vector_db, llm)
            return "没有找到新文档，已使用现有数据重新加载 RAG 链。"
        else:
            # No documents AND no existing DB
            return "错误：没有文档可加载，且没有现有的向量数据库。"

    # Step 2: Split text
    print("分割文本...")
    chunks = split_documents(documents)
    if not chunks:
        print("分割后未生成文本块。")
        # Try loading existing DB if splitting yielded nothing
        print("尝试加载现有向量数据库...")
        vector_db = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db:
             print("警告：新加载的文档分割后未产生任何文本块。使用现有数据库。")
             rag_chain = create_rag_chain(vector_db, llm) # Ensure chain is recreated
             return "警告：文档分割后未产生任何文本块。RAG 链已使用现有数据重新加载。"
        else:
            # No chunks AND no existing DB
            return "错误：文档分割后未产生任何文本块，且无现有数据库。"

    # Step 3: Load or Create/Update vector database
    print("加载或更新向量数据库...")
    # Try loading first, even if we have chunks (in case we want to add to it)
    vector_db_loaded = get_vector_db(None, embeddings, PERSIST_DIR)

    if vector_db_loaded:
        print(f"向现有向量数据库添加 {len(chunks)} 个块...")
        vector_db = vector_db_loaded # Use the loaded DB
        try:
            # Consider adding only new chunks if implementing duplicate detection later
            vector_db.add_documents(chunks)
            print("块添加成功。")
            # Persisting might be needed depending on Chroma version/setup, often automatic.
            # vector_db.persist() # Uncomment if persistence issues occur
        except Exception as e:
             print(f"添加文档到 Chroma 时出错: {e}")
             # If adding fails, proceed with the DB as it was before adding
             rag_chain = create_rag_chain(vector_db, llm)
             return f"错误：向向量数据库添加文档时出错: {e}。RAG链可能使用旧数据。"
    else:
        # Database didn't exist or couldn't be loaded, create a new one with the current chunks
        print(f"创建新的向量数据库并添加 {len(chunks)} 个块...")
        try:
            # Call get_vector_db again, this time *with* chunks to trigger creation
            vector_db = get_vector_db(chunks, embeddings, PERSIST_DIR)
            if vector_db is None: # Check if creation failed within get_vector_db
                 raise RuntimeError("get_vector_db failed to create a new database.")
            print("新的向量数据库已创建并持久化。")
        except Exception as e:
            print(f"创建新的向量数据库时出错: {e}")
            return f"错误：创建向量数据库失败: {e}"

    if vector_db is None:
         # This should ideally not be reached if error handling above is correct
         return "错误：未能加载或创建向量数据库。"

    # Step 4: Create RAG chain
    print("创建 RAG 链...")
    rag_chain = create_rag_chain(vector_db, llm)
    print("索引和 RAG 链已成功更新。")
    return "文档处理完成，索引和 RAG 链已更新。"

# Helper function to list documents in the directory
def get_loaded_documents_list():
    """Returns a Markdown formatted list of files in DOCUMENTS_DIR."""
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        return "当前没有已加载的文档。"
    try:
        files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f)) and (f.endswith('.pdf') or f.endswith('.docx') or f.endswith('.doc'))]
        if not files:
            return "当前没有已加载的文档。"
        markdown_list = "### 当前已加载文档:\n" + "\n".join([f"- {file}" for file in files])
        return markdown_list
    except Exception as e:
        print(f"Error listing documents: {e}")
        return "无法列出文档。"


# 9. Function to handle file uploads (Modified to return doc list)
def handle_file_upload(file_obj):
    """Saves the uploaded file, triggers index rebuilding, and returns status and doc list."""
    if file_obj is None:
        return "未选择文件。", get_loaded_documents_list() # Return current list even if no file selected

    try:
        # Gradio provides a temporary file path
        temp_file_path = file_obj.name
        file_name = os.path.basename(temp_file_path)
        destination_path = os.path.join(DOCUMENTS_DIR, file_name)

        print(f"将上传的文件从 {temp_file_path} 复制到 {destination_path}")
        # Ensure documents directory exists
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
        shutil.copy(temp_file_path, destination_path) # Copy the file

        print(f"文件 {file_name} 上传成功。开始重建索引...")
        status = rebuild_index_and_chain()
        final_status = f"文件 '{file_name}' 上传成功。\n{status}"
        # Get updated document list
        doc_list_md = get_loaded_documents_list()
        return final_status, doc_list_md

    except Exception as e:
        print(f"文件上传或处理失败: {e}")
        # Return error and current doc list
        return f"文件上传或处理失败: {e}", get_loaded_documents_list()

# Updated function to handle query submission for gr.Chatbot
def handle_submit_with_thinking(query_text, chat_history):
    if not query_text or query_text.strip() == "":
        # Optionally, add a message to chat_history if query is empty, or do nothing
        # chat_history.append((None, "请输入问题。"))
        yield "", chat_history # Clear input, return current history
        return

    chat_history.append((query_text, "思考中..."))
    yield "", chat_history # Clear input, update history with user query and "Thinking..."

    # This variable will hold the latest full response from the RAG chain
    # as process_query yields accumulated responses.
    final_response_from_rag = "思考中..." 

    for stream_chunk in process_query(query_text): # process_query yields full accumulated answer
        final_response_from_rag = stream_chunk
        chat_history[-1] = (query_text, final_response_from_rag) # Update the AI's part of the last message
        yield "", chat_history
    
    # If process_query finishes and the last message was still "思考中..."
    # (e.g., if process_query yielded an error or empty response immediately)
    # ensure it's updated. However, process_query should yield a proper final message.
    if chat_history and chat_history[-1][1] == "思考中...":
        # This case implies process_query might not have yielded a replacement.
        # If final_response_from_rag is still "思考中...", it means process_query didn't provide a different final state.
        # This should ideally be handled by process_query yielding a specific error or "no answer" message.
        # For safety, if final_response_from_rag is still "思考中...", we can set a generic error.
        # However, current process_query yields "抱歉，未能生成回答。" which is good.
        pass


# 10. 主函数 (Modified for Gradio Blocks, Upload, Doc List, Streaming, and Usage Guide)
def main():
    global embeddings, llm, rag_chain # Declare globals needed

    print(f"IMPORTANT: Current embedding model is {EMBEDDING_MODEL_PATH}.")
    print(f"If you recently changed the embedding model and encounter dimension mismatch errors,")
    print(f"you MUST manually delete the ChromaDB directory: '{PERSIST_DIR}' and restart.")

    # Ensure documents directory exists at start
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"创建文档目录: {DOCUMENTS_DIR}")
        print("请将您的 PDF 和 DOCX 文件添加到此目录或使用上传功能。")

    # Initialize embeddings and LLM once
    print("初始化 Embedding 模型...")
    embeddings = initialize_embeddings()

    print("初始化 LLM 客户端...")
    llm = initialize_openai_client()

    # Initial index and chain build
    print("执行初始索引构建...")
    initial_status = rebuild_index_and_chain()
    print(initial_status)
    if rag_chain is None and "错误" in initial_status:
        print("无法初始化 RAG 链。请检查文档或配置。退出。")
        # Optionally, allow Gradio to launch but show an error state
        # return # Or let Gradio launch to show the error

    # Get initial document list
    initial_doc_list = get_loaded_documents_list()

    # --- Custom CSS for ChatGPT-like styling ---
    # Base styling - can be expanded significantly
    custom_css = """
body, .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { background-color: #F7F7F8; } /* Light background */

/* Chatbot styling */
.gr-chatbot { border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
.gr-chatbot .message-wrap { box-shadow: none !important; } /* Remove default shadow on messages */
.gr-chatbot .message.user { background-color: #FFFFFF; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; margin-left: auto; max-width: 70%;}
.gr-chatbot .message.bot { background-color: #F7F7F8; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; max-width: 70%;}
.gr-chatbot .message.bot.thinking { color: #888; font-style: italic; } /* Style for "Thinking..." */

/* Input area styling */
#chat_input_row { /* Style for the Row containing input and button */
    display: flex !important;
    align-items: center !important; /* Vertically align items (textbox and button) */
    gap: 8px !important; /* Add a small gap between textbox and button */
}
#chat_input_row .gr-textbox textarea { 
    border-radius: 18px !important; 
    border: 1px solid #E0E0E0 !important; 
    padding: 12px 15px !important; 
    font-size: 1rem !important;
    background-color: #FFFFFF !important;
    box-sizing: border-box !important; /* Ensure padding and border are part of the element's total width and height */
    line-height: 1.4 !important; /* Consistent line height */
    min-height: 46px !important; /* Ensure a minimum height, helps with single line consistency */
}
#chat_input_row .gr-button { 
    border-radius: 18px !important; 
    font-weight: 500 !important;
    background-color: #10A37F !important; /* ChatGPT-like green */
    color: white !important; 
    border: none !important;
    min-width: 80px !important;
    font-size: 1rem !important; /* Match textarea font size */
    /* Textarea has 12px padding + 1px border = 13px effective 'outer' space top/bottom. */
    /* Button has no border, so its padding should be 13px top/bottom. */
    padding: 13px 15px !important; 
    box-sizing: border-box !important; /* Ensure padding is part of the element's total width and height */
    line-height: 1.4 !important; /* Match textarea line height */
    height: 46px !important; /* Explicit height to match textarea's typical single-line height */
}
#chat_input_row .gr-button:hover { background-color: #0F8E6C !important; }

/* General Tab Styling */
.tab-nav button { border-radius: 8px 8px 0 0 !important; padding: 10px 15px !important; }
.tab-nav button.selected { background-color: #E0E0E0 !important; border-bottom: 2px solid #10A37F !important;}
.gr-panel { background-color: #FFFFFF; border-radius: 0 0 8px 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); } /* Panel for tab content */
    """

    # --- Gradio Interface using Blocks ---
    print("\n设置 Gradio 界面...")
    with gr.Blocks(theme=gr_themes.Soft(), css=custom_css) as iface:
        gr.Markdown(f"""
        <div style='text-align: center;'>
        <h1>耀安科技-煤矿大模型知识问答系统</h1>
        <p>根据已有的文档或您上传的文档提问。</p>
        </div>
        """)

        with gr.Tab("问答"):
            with gr.Column(elem_id="chat-column"): # Added a column for better layout control
                chatbot_output = gr.Chatbot(
                    label="对话窗口",
                    bubble_full_width=False, # Bubbles don't take full width
                    height=600, # Set a fixed height for the chat area
                    avatar_images=(None, "https://img.icons8.com/fluency/48/chatbot.png"), # User avatar none, bot has a simple icon
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False}
                    ]
                    # render_markdown=True,  # Explicitly set, default is True
                    # sanitize_html=False    # Test by disabling HTML sanitization
                )
                with gr.Row(elem_id="chat_input_row"): # Row for input textbox and button
                    query_input = gr.Textbox(
                        show_label=False,
                        placeholder="在此输入您的问题...",
                        lines=1, # Single line input initially, can expand
                        scale=4 # Textbox takes more space
                    )
                    submit_button = gr.Button("发送", scale=1) # "Send" button

        with gr.Tab("上传与管理文档"): # Renamed tab for clarity
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="上传 PDF 或 DOCX 文件", file_types=['.pdf', '.docx', '.doc'])
                    upload_button = gr.Button("上传并重建索引")
                    upload_status = gr.Textbox(label="上传状态", interactive=False)
                with gr.Column(scale=1):
                    # Component to display loaded documents
                    loaded_docs_display = gr.Markdown(value=initial_doc_list)

        with gr.Tab("使用教程"):
            gr.Markdown("""
            ## 如何使用本 RAG 系统

            **1. 准备文档:**
               - 您可以将包含知识的 PDF 或 Word 文档（.pdf, .docx, .doc）放入程序运行目录下的 `documents` 文件夹中。
               - 程序启动时会自动加载 `documents` 文件夹中的所有支持的文档。

            **2. 上传文档:**
               - 切换到 **上传与管理文档** 标签页。
               - 点击“浏览文件”按钮选择您想要上传的 PDF 或 Word 文档。
               - 点击 **上传并重建索引** 按钮。系统会将文件复制到 `documents` 目录，并更新知识库。
               - 上传和处理需要一些时间，请耐心等待状态更新。
               - 右侧会显示当前 `documents` 目录中已加载的文件列表。

            **3. 提问:**
               - 切换到 **问答** 标签页。
               - 在 **问题** 输入框中输入您想基于文档内容提出的问题。
               - 点击 **提交问题** 按钮或按 Enter 键。
               - 系统将根据文档内容检索相关信息，并使用大语言模型（Qwen 3-8B）生成回答。
               - 回答将在 **回答** 框中以流式方式显示。

            **注意:**
               - 重建索引可能需要一些时间，特别是对于大型文档或大量文档。
               - 回答的质量取决于文档内容的相关性和模型的理解能力。
               - 目前系统每次上传文件后会重新处理 `documents` 目录下的 *所有* 文件。对于非常大的知识库，未来可能需要优化为仅处理新增文件。
            """)


        # --- Event Handlers ---
        # Q&A Submission for Chatbot
        # The `fn` now takes query_input and chatbot_output (history)
        # It returns a tuple: (new_value_for_query_input, new_value_for_chatbot_output)
        submit_button.click(
            fn=handle_submit_with_thinking,
            inputs=[query_input, chatbot_output],
            outputs=[query_input, chatbot_output] # query_input is cleared, chatbot_output is updated
        )
        query_input.submit( 
             fn=handle_submit_with_thinking,
             inputs=[query_input, chatbot_output],
             outputs=[query_input, chatbot_output]
        )

        # File Upload and Rebuild
        upload_button.click(
            fn=handle_file_upload,
            inputs=file_input,
            outputs=[upload_status, loaded_docs_display] # Update both status and doc list
        )

    print("启动 Gradio 界面...")
    # Launch the interface
    iface.launch(server_name="0.0.0.0") # Listen on all interfaces

if __name__ == "__main__":
    main()