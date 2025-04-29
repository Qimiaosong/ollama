"""这两行是从 langchain_community（LangChain 的社区扩展模块）
中导入了两种 PDF 加载器
"""
# 用于读取本地 未结构化 的 PDF 文件内容
from langchain_community.document_loaders import UnstructuredPDFLoader
# 用于从 线上 URL 加载 PDF 文件（虽然这里没用到）
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "./data/BOI.pdf" # PDF文件在本地的路径
model = "llama3.2" # 本代码中使用的大语言模型

# 加载本地文件
if doc_path:
    # 创建一个PDF加载器对象
    loader = UnstructuredPDFLoader(file_path=doc_path)
    # 执行加载，得到的是一个文档列表（LangChain中的Document对象）
    data = loader.load()
    print("done loading....")
else:
    # 如果没有本地文件，使用在线加载器
    print("Upload a PDF file")

# 获取该文档片段的纯文本内容，也就是 PDF 第一页的文字
content = data[0].page_content
# print(content[:100])

"""
将pdf文档切分成小块，并使用本地Ollama模型生成向量嵌入，
将其存入Chroma向量数据库中，为后续RAG问答作准备
"""
# 用于从本地 Ollama 模型（如 nomic-embed-text）中生成文本嵌入向量。
from langchain_ollama import OllamaEmbeddings
# RecursiveCharacterTextSplitter: 一个智能的文本切分器，保持语义连续性的同时将长文档拆成小段
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 一个轻量级、易本地部署的向量数据库
from langchain_community.vectorstores import Chroma

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
# data 是从 PDF 中提取的 Document 列表，执行切块操作。
chunks = text_splitter.split_documents(data)
print("done splitting....")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

# ===== Add to vector database ===
import ollama
# 使用 pull() 下载嵌入模型 nomic-embed-text，这是 Ollama 上可以跑的 text embedding 模型
ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks, # 向量化的数据来源，是你切好的文本片段
    embedding=OllamaEmbeddings(model="nomic-embed-text"), # 指定使用 nomic-embed-text 模型来计算每段文本的嵌入向量
    collection_name="simple-rag", # 向量数据库的名称
)
# 你创建了一个本地的 RAG-ready 向量数据库（用 Chroma 实现），存放了你 PDF 的语义向量表示
print("done adding to vector database....")

"""
这段代码是构建 RAG（Retrieval-Augmented Generation）
问答系统的完整流程的**“检索 + 生成”部分**，核心功能是：
💡用户提问 → 检索相关文档片段 → 用大语言模型回答问题。
"""
# LangChain 用来构建提示词模版的模块
from langchain.prompts import ChatPromptTemplate, PromptTemplate
# 用于解析模型输出为字符串
from langchain_core.output_parsers import StrOutputParser
# 用来调用你本地的 Ollama 模型进行聊天
from langchain_ollama import ChatOllama
# LangChain 的“原样传递”组件
from langchain_core.runnables import RunnablePassthrough
# LangChain 的一个增强检索器，可以基于一个问题生成多个相似问题来扩大召回
from langchain.retrievers.multi_query import MultiQueryRetriever

"""
加载你本地 Ollama 环境中的模型
这个模型用于两件事：
生成多个问题（MultiQueryRetriever 中用）
最终回答问题（RAG 中用）
"""
llm = ChatOllama(model=model)

# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
# 把一个问题扩展成多个“变体”，然后分别去向量数据库中检索相关内容，再合并结果。
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)


# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
# 这是传给模型的提示词结构,指示模型“只能根据上下文来回答问题"
prompt = ChatPromptTemplate.from_template(template)

# 构建完整的 RAG Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# res = chain.invoke(input=("what is the document about?",))
# res = chain.invoke(
#     input=("what are the main points as a business owner I should be aware of?",)
# )
# 调用问答链
res = chain.invoke(input=("how to report BOI?",))

print(res)