"""
Gradio UI interface for the RAG system.
"""
from typing import Tuple, List, Iterator
import gradio as gr
import gradio.themes as gr_themes
from pathlib import Path

from config.settings import settings
from src.rag_system.core.rag_system import RAGSystem
from src.rag_system.utils.logger import get_logger

logger = get_logger()


class RAGInterface:
    """Gradio interface for the RAG system."""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.interface = None
        self._setup_interface()
    
    def _setup_interface(self) -> None:
        """Setup the Gradio interface."""
        try:
            logger.info("Setting up Gradio interface...")
            
            # Custom CSS for modern styling
            custom_css = self._get_custom_css()
            
            with gr.Blocks(theme=gr_themes.Soft(), css=custom_css, title=settings.APP_NAME) as interface:
                # Header
                gr.Markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                <h1 style='margin-bottom: 10px;'>{settings.APP_NAME}</h1>
                <p style='color: #666; font-size: 16px;'>{settings.APP_DESCRIPTION}</p>
                <p style='color: #888; font-size: 14px;'>版本 {settings.APP_VERSION}</p>
                </div>
                """)
                
                # Main content tabs
                with gr.Tab("💬 智能问答"):
                    self._create_chat_tab()
                
                with gr.Tab("📁 文档管理"):
                    self._create_document_tab()
                
                with gr.Tab("ℹ️ 系统状态"):
                    self._create_status_tab()
                
                with gr.Tab("📖 使用说明"):
                    self._create_help_tab()
            
            self.interface = interface
            logger.info("Gradio interface setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup Gradio interface: {str(e)}")
            raise
    
    def _create_chat_tab(self) -> None:
        """Create the chat interface tab."""
        with gr.Column(elem_id="chat-column"):
            # Chat interface
            chatbot_output = gr.Chatbot(
                label="对话历史",
                height=600,
                bubble_full_width=False,
                avatar_images=(None, "🤖"),
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False}
                ]
            )
            
            # Input area
            with gr.Row(elem_id="chat_input_row"):
                query_input = gr.Textbox(
                    show_label=False,
                    placeholder="在此输入您的问题...",
                    lines=1,
                    scale=4,
                    container=False
                )
                submit_button = gr.Button("发送", scale=1, variant="primary")
            
            # Clear button
            clear_button = gr.Button("清空对话", variant="secondary")
            
            # Event handlers
            submit_button.click(
                fn=self._handle_chat_submit,
                inputs=[query_input, chatbot_output],
                outputs=[query_input, chatbot_output]
            )
            
            query_input.submit(
                fn=self._handle_chat_submit,
                inputs=[query_input, chatbot_output],
                outputs=[query_input, chatbot_output]
            )
            
            clear_button.click(
                fn=lambda: ("", []),
                outputs=[query_input, chatbot_output]
            )
    
    def _create_document_tab(self) -> None:
        """Create the document management tab."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 上传文档")
                file_input = gr.File(
                    label="选择文档文件",
                    file_types=settings.SUPPORTED_FILE_EXTENSIONS,
                    file_count="single"
                )
                upload_button = gr.Button("上传并处理", variant="primary")
                upload_status = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=3
                )
                
                gr.Markdown("### 系统操作")
                rebuild_button = gr.Button("重建索引", variant="secondary")
                rebuild_status = gr.Textbox(
                    label="重建状态",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 已加载文档")
                loaded_docs_display = gr.Markdown(
                    value=self.rag_system.get_document_list_markdown()
                )
                refresh_docs_button = gr.Button("刷新列表", variant="secondary")
        
        # Event handlers
        upload_button.click(
            fn=self._handle_file_upload,
            inputs=file_input,
            outputs=[upload_status, loaded_docs_display]
        )
        
        rebuild_button.click(
            fn=self._handle_rebuild_index,
            outputs=[rebuild_status, loaded_docs_display]
        )
        
        refresh_docs_button.click(
            fn=self.rag_system.get_document_list_markdown,
            outputs=loaded_docs_display
        )
    
    def _create_status_tab(self) -> None:
        """Create the system status tab."""
        with gr.Column():
            gr.Markdown("### 系统状态监控")
            
            status_display = gr.JSON(
                label="系统状态",
                value=self.rag_system.get_system_status()
            )
            
            refresh_status_button = gr.Button("刷新状态", variant="primary")
            
            refresh_status_button.click(
                fn=self.rag_system.get_system_status,
                outputs=status_display
            )
    
    def _create_help_tab(self) -> None:
        """Create the help/documentation tab."""
        help_content = f"""
        ## 📖 {settings.APP_NAME} 使用指南

        ### 🚀 系统概述
        本系统基于检索增强生成（RAG）技术，能够根据您上传的文档内容回答问题。系统会自动分析文档，建立知识索引，并结合大语言模型生成准确的答案。

        ### 📋 支持的文档格式
        - **PDF文件** (.pdf)
        - **Word文档** (.docx, .doc)
        - **文本文件** (.txt)

        ### 💡 使用步骤

        #### 1. 📁 上传文档
        - 切换到 **文档管理** 标签页
        - 点击"选择文档文件"按钮，选择您要上传的文档
        - 点击 **上传并处理** 按钮
        - 系统将自动处理文档并更新知识库

        #### 2. 💬 智能问答
        - 切换到 **智能问答** 标签页
        - 在输入框中输入您的问题
        - 点击 **发送** 按钮或按回车键
        - 系统将基于已上传的文档内容生成回答

        #### 3. 🔧 系统管理
        - 在 **文档管理** 页面可以查看已加载的文档列表
        - 使用 **重建索引** 功能重新处理所有文档
        - 在 **系统状态** 页面监控系统运行状态

        ### ⚠️ 注意事项

        - **文档质量**：上传高质量、结构化的文档可以获得更好的问答效果
        - **处理时间**：大型文档的处理需要一定时间，请耐心等待
        - **问题表述**：清晰、具体的问题能够获得更准确的答案
        - **系统资源**：确保有足够的GPU/CPU资源用于模型推理

        ### 🔧 技术参数

        - **embedding模型**: {settings.EMBEDDING_MODEL_PATH}
        - **LLM模型**: {settings.VLLM_MODEL_NAME}
        - **文本块大小**: {settings.CHUNK_SIZE}
        - **检索文档数**: {settings.SEARCH_K}

        ### 📞 技术支持

        如遇到问题，请检查：
        1. VLLM服务是否正常运行
        2. embedding模型路径是否正确
        3. 系统日志中的错误信息

        ---
        *{settings.APP_NAME} v{settings.APP_VERSION}*
        """
        
        gr.Markdown(help_content)
    
    def _handle_chat_submit(self, query: str, chat_history: List) -> Tuple[str, List]:
        """Handle chat message submission."""
        try:
            if not query or not query.strip():
                return "", chat_history
            
            # Add user message to history
            chat_history.append((query, "思考中..."))
            
            # Generate response
            full_response = ""
            for response_chunk in self.rag_system.process_query(query):
                full_response = response_chunk
                # Update the last message in history
                chat_history[-1] = (query, full_response)
                yield "", chat_history
            
            # Ensure we have a final response
            if not full_response or full_response == "思考中...":
                chat_history[-1] = (query, "抱歉，未能生成回答。请检查系统状态或重新表述问题。")
            
            return "", chat_history
            
        except Exception as e:
            logger.error(f"Error in chat submit: {str(e)}")
            if chat_history:
                chat_history[-1] = (query, f"处理消息时发生错误: {str(e)}")
            else:
                chat_history.append((query, f"处理消息时发生错误: {str(e)}"))
            return "", chat_history
    
    def _handle_file_upload(self, file_obj) -> Tuple[str, str]:
        """Handle file upload."""
        try:
            if file_obj is None:
                return "未选择文件。", self.rag_system.get_document_list_markdown()
            
            # Get file info
            file_path = Path(file_obj.name)
            file_name = file_path.name
            
            logger.info(f"Processing uploaded file: {file_name}")
            
            # Process the file
            status, doc_list = self.rag_system.upload_and_process_file(file_path, file_name)
            
            # Return formatted document list
            doc_list_md = self.rag_system.get_document_list_markdown()
            
            return status, doc_list_md
            
        except Exception as e:
            error_msg = f"文件上传处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, self.rag_system.get_document_list_markdown()
    
    def _handle_rebuild_index(self) -> Tuple[str, str]:
        """Handle index rebuild."""
        try:
            logger.info("User requested index rebuild")
            status = self.rag_system.rebuild_index()
            doc_list_md = self.rag_system.get_document_list_markdown()
            return status, doc_list_md
            
        except Exception as e:
            error_msg = f"重建索引失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, self.rag_system.get_document_list_markdown()
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        /* Global Styles */
        .gradio-container {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Header Styles */
        .main-header {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            margin-bottom: 24px;
        }
        
        /* Chat Interface */
        #chat-column {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }
        
        /* Chat Input */
        #chat_input_row {
            margin-top: 16px;
            gap: 12px;
        }
        
        #chat_input_row .gr-textbox {
            border-radius: 24px;
            border: 2px solid #e1e5e9;
            transition: border-color 0.2s ease;
        }
        
        #chat_input_row .gr-textbox:focus-within {
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        
        #chat_input_row .gr-button {
            border-radius: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px 24px;
            transition: transform 0.2s ease;
        }
        
        #chat_input_row .gr-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Chatbot Messages */
        .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 18px 18px 4px 18px;
            padding: 12px 16px;
            margin: 8px 0;
            margin-left: auto;
            max-width: 80%;
        }
        
        .message.bot {
            background: #f8f9fa;
            color: #333;
            border: 1px solid #e9ecef;
            border-radius: 18px 18px 18px 4px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
        }
        
        /* Tabs */
        .tab-nav button {
            border-radius: 8px 8px 0 0;
            padding: 12px 20px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .tab-nav button:hover {
            background-color: #f8f9fa;
        }
        
        .tab-nav button.selected {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom: none;
        }
        
        /* Buttons */
        .gr-button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .gr-button.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
        }
        
        .gr-button.secondary {
            background: #6c757d;
            color: white;
            border: none;
        }
        
        .gr-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        /* Cards and Panels */
        .gr-panel {
            background: white;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        /* File Upload */
        .gr-file {
            border: 2px dashed #ccc;
            border-radius: 12px;
            transition: all 0.2s ease;
        }
        
        .gr-file:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
        
        /* Status Indicators */
        .status-success {
            color: #28a745;
            font-weight: 500;
        }
        
        .status-error {
            color: #dc3545;
            font-weight: 500;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: 500;
        }
        """
    
    def launch(self) -> None:
        """Launch the Gradio interface."""
        try:
            if not self.interface:
                raise ValueError("Interface not initialized")
            
            logger.info("Launching Gradio interface...")
            logger.info(f"Server will be available at: http://{settings.GRADIO_SERVER_NAME}:{settings.GRADIO_SERVER_PORT or 'auto'}")
            
            self.interface.launch(
                server_name=settings.GRADIO_SERVER_NAME,
                server_port=settings.GRADIO_SERVER_PORT,
                share=settings.GRADIO_SHARE,
                show_error=True,
                quiet=False
            )
            
        except Exception as e:
            logger.error(f"Failed to launch Gradio interface: {str(e)}")
            raise
