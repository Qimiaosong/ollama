import pprint
import ollama
import os
import datetime

from langchain_community.document_loaders import PDFPlumberLoader

model = "llama3.2"

pdf_files = [f for f in os.listdir("./data") if f.endswith(".pdf")]

# 存储所有pdf文档内容
all_pages = []

for pdf_file in pdf_files:

    file_path = os.path.join("./data", pdf_file)
    print(f"Processing PDF file: {pdf_file}")

    # 使用PDFPlumberLoader加载PDF文件
    loader = PDFPlumberLoader(file_path=file_path)
    # 将文件分割成多个页面
    pages = loader.load_and_split()
    print(f"pages length: {len(pages)}")

    all_pages.extend(pages)

    # 提取第一页的内容
    text = pages[0].page_content
    print(f"Text extracted from the PDF file '{pdf_file}':\n{text}\n")

    # 给模型构建提示词
    prompt = f"""
    You are an AI assistant that helps with summarizing PDF documents.
    
    Here is the content of the PDF file '{pdf_file}':
    
    {text}
    
    Please summarize the content of this document in a few sentences.
    """

    # Send the prompt and get the response
    try:
        response = ollama.generate(model=model, prompt=prompt)
        summary = response.get("response", "")

        # print(f"Summary of the PDF file '{pdf_file}':\n{summary}\n")
    except Exception as e:
        print(
            f"An error occurred while summarizing the PDF file '{pdf_file}': {str(e)}"
        )

# 使用LangChain的RecursiveCharacterTextSplitter将长文本分割成较小的块
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

# 分块后的文本
text_chunks = []
# 遍历之前加载的所有PDF页面(all_pages)，对每页内容进行分块
for page in all_pages:
    chunks = text_splitter.split_text(page.page_content)
    text_chunks.extend(chunks)

print(f"Number of text chunks: {text_chunks}")


# 为每个文本块添加结构化元数据信息
def add_metadata(chunks, doc_title):
    metadata_chunks = []
    for chunk in chunks:
        metadata = {
            "title": doc_title,
            "author": "US Business Bureau",  # Update based on document data
            "date": str(datetime.date.today()),
        }
        metadata_chunks.append({"text": chunk, "metadata": metadata})
    return metadata_chunks

# 带元数据的文本块
metadata_text_chunks = add_metadata(text_chunks, "BOI US FinCEN")
# pprint.pprint(f"metadata text chunks: {metadata_text_chunks}")


# 专门优化的嵌入模型nomic-embed-text将文本转为向量
ollama.pull("nomic-embed-text")

def generate_embeddings(text_chunks, model_name="nomic-embed-text"):
    # 文本嵌入向量
    embeddings = []
    for chunk in text_chunks:
        embedding = ollama.embeddings(model=model_name, prompt=chunk)
        embeddings.append(embedding)
    return embeddings

texts = [chunk["text"] for chunk in metadata_text_chunks]
embeddings = generate_embeddings(texts)
print(f"Embeddings: {embeddings}")


# 构建向量数据库,将之前处理好的文本块和元数据存储到Chroma向量数据库中
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings

# 将之前处理的metadata_text_chunks转换为LangChain的标准Document格式
docs = [
    Document(page_content=chunk["text"], metadata=chunk["metadata"])
    for chunk in metadata_text_chunks
]


from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# 高性能嵌入模型
fastembedding = FastEmbedEmbeddings()
# 指定向量数据库的本地存储路径
vector_db_path = "./db/vector_db"

vector_db = Chroma.from_documents(
    documents=docs, # 要存储的Document对象列表
    embedding=fastembedding, # 使用的嵌入模型（FastEmbed）
    persist_directory=vector_db_path, # 数据库持久化存储路径
    # embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="docs-local-rag",
)


from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# LLM from Ollama
local_model = "llama3.2"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)


retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = """
by when should I file if my business was established in 2013?"""
print((chain.invoke(questions)))
response = chain.invoke(questions)

# # 使用 ElevenLabs 的 API 将文本转换为语音并播放
from elevenlabs.client import ElevenLabs # 用于与 ElevenLabs API 交互
from elevenlabs import play # 用于一次性播放生成的音频
from elevenlabs import stream # 用于流式播放音频
from dotenv import load_dotenv # 用于加载环境变量

load_dotenv()

text_response = response

api_key = os.getenv("ELEVENLABS_API_KEY")

# # 调用 ElevenLabs API 生成语音
client = ElevenLabs(api_key=api_key)
audio_stream = client.generate(text=text_response, model="eleven_turbo_v2", stream=True)
# play 会等待整个音频生成完毕后再播放（代码中被注释掉了）
# play(audio_stream)
# 使用 stream 函数流式播放生成的音频，会边生成边播放，延迟更低
stream(audio_stream)