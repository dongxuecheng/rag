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

# --- Configuration ---
DOCUMENTS_DIR = "./documents"  # Modify to your document directory
PERSIST_DIR = "./chroma_db"     # Vector database storage directory
EMBEDDING_MODEL_PATH = "/home/dxc/model/bge-m3"
EMBEDDING_DEVICE = "cuda:1" # Or 'cpu'
# VLLM Server details (using OpenAI compatible endpoint)
VLLM_BASE_URL = "http://localhost:8000/v1" # Default for `vllm serve`
VLLM_API_KEY = "dummy-key" # Required by ChatOpenAI, but VLLM server doesn't usually check it
VLLM_MODEL_NAME = "/home/dxc/model/qwen2.5-7b" # The model name used in `vllm serve`
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 3 # Number of documents to retrieve
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
        length_function=len
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
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception as e:
            print(f"Error loading existing vector database: {e}. Will attempt to create a new one if chunks are provided.")
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
    retriever = vector_db.as_retriever(search_kwargs={"k": SEARCH_K})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain

# 7. Function to process query using the RAG chain (Modified for Streaming)
def process_query(query):
    """Processes a user query using the RAG chain and streams the answer."""
    global rag_chain
    if rag_chain is None:
        yield "错误：RAG 链未初始化。"
        return

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

# 10. 主函数 (Modified for Gradio Blocks, Upload, Doc List, Streaming, and Usage Guide)
def main():
    global embeddings, llm, rag_chain # Declare globals needed

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

    # --- Gradio Interface using Blocks ---
    print("\n设置 Gradio 界面...")
    with gr.Blocks() as iface:
        gr.Markdown(f"""
        <h1 style='text-align: center; margin-bottom: 1rem'>简单 RAG 问答系统</h1>
        <p style='text-align: center; font-size: 1rem; color: grey;'>根据已有的文档或您上传的文档提问。
        使用Qwen 2.5-7B模型进行问答。支持流式回答。</p>

        """)

        with gr.Tab("问答"):
            # Textbox supports streaming updates when the function yields
            chatbot_output = gr.Textbox(label="回答", interactive=False, lines=10)
            query_input = gr.Textbox(lines=2, placeholder="在此输入您的问题...", label="问题")
            submit_button = gr.Button("提交问题")

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
               - 系统将根据文档内容检索相关信息，并使用大语言模型（Qwen 2.5-7B）生成回答。
               - 回答将在 **回答** 框中以流式方式显示。

            **注意:**
               - 重建索引可能需要一些时间，特别是对于大型文档或大量文档。
               - 回答的质量取决于文档内容的相关性和模型的理解能力。
               - 目前系统每次上传文件后会重新处理 `documents` 目录下的 *所有* 文件。对于非常大的知识库，未来可能需要优化为仅处理新增文件。
            """)


        # --- Event Handlers ---
        # Q&A Submission (No change needed here, Gradio handles generators for streaming)
        submit_button.click(
            fn=process_query,
            inputs=query_input,
            outputs=chatbot_output
        )
        query_input.submit( # Allow pressing Enter to submit
             fn=process_query,
             inputs=query_input,
             outputs=chatbot_output
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