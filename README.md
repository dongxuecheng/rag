# rag

## 环境搭建
### miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
安装过程中选择yes，使其每次启动shell都运行conda环境
### miniconda清华源配置
```bash
conda config --set show_channel_urls yes
```
修改.condarc文件内容
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
清除索引
```bash
conda clean -i
```
### pip清华源配置
```bash
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
### VLLM安装
```bash
# (Recommended) Create a new conda environment.
conda create -n rag python=3.12 -y
conda activate rag
pip install vllm
```
### Huggingface国内镜像源配置
在文件末尾添加
```bash
nano ~/.bashrc 
export HF_ENDPOINT=https://hf-mirror.com
source ~/.bashrc
```

### Huggingface 下载模型

```bash
#千问模型作为llm
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/model/qwen2.5-7b --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen3-8B --local-dir ~/model/qwen3-8b --local-dir-use-symlinks False

#Embedding模型
huggingface-cli download BAAI/bge-m3 --local-dir ~/model/bge-m3 --local-dir-use-symlinks False
huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir ~/model/get-qwen2-1.5b --local-dir-use-symlinks False
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ~/model/all-minilm-l6 --local-dir-use-symlinks False
```

### 安装langchain相关包

```bash
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain_openai
pip install pypdf
pip install docx2txt
pip install gradio
```
### 启动VLLM

```bash
vllm serve ~/model/qwen2.5-7b
vllm serve /mnt/dxc/model/qwen3-8b --enable-reasoning --reasoning-parser deepseek_r1 --max-model-len 16384
```