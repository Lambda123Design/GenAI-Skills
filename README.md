# GenAI-Skills

**Table of Contents:**

**A) Langchain and OpenAI**

**B) Data Ingestion with Document Loaders**

**C) Text Splitter Techniques - Recursive Character Text Splitter**

**D) Text Splitting Technique - Character Text Splitter**

**E) Text Splitting Technique - HTML Header Text Splitter**

**F) Recursive JSON Splitter**

**G) Introduction to OpenAI Embedding**

**H) Ollama Embeddings**

**I) HuggingFace Embeddings**

**J) VectorStore - FAISS**

**K) VectorStore - ChromaDB**

**L) Building Important Components of Langchain**

**M) Creating Application using Langchain**

**N) Understanding Retrievers and Chains**

**O) Introduction to Ollama and Setup**

**P) GenAI Application using Ollama**

**Q) Tracking GenAI Application using LangSmith**

**R) Building Gen AI App using LCEL - Getting Started with Open Source Models using Groq API**

**S) Building Gen AI App using LCEL - Building LLM, Prompt and StrOuput Chains with LCEL**

**T) Building Gen AI App using LCEL - Deploy LangServe Runnable and Chain as API**

**VII) Building Chatbots With Conversation History Using Langchain**

**A) Building Chatbot With Message History Using Langchain**

**B) Working With Prompt Termplate And Message Chat History Using Langchain**

**C) Managing the Chat Conversation History Using Langchain**

**D) Working With VectorStore And Retriever**

**VIII) Conversational Q&A Chatbot With Message History**

**A) Building Conversational Q&A Chatbot With Message History**

**IX)  End To End Q&A Chatbot GEN AI App With**

**A) Introduction To The Q&A Chatbot**

**B) Creating Virtual Environment**

**C) Creating Prompt Template And Integrating Open AI API**

**D) Creating Streamlit Web App and Integrating Response With OpenAI API**

**E) Q&A Chatbot With Ollama And Open Source Models**

**X) RAG Document Q&A With GROQ API And LLama3**

**A) Introduction To Groq Cloud And LPU Inference Engine**

**B) RAG Document Q&A With GROQ API And LLama3**

**XI) Conversational Q&A Chatbot- Chat With Pdf Along With Chat History**

**A) Demo of the Conversational Q&A Chatbot**

**B) End To End Conversational Q&A Chatbot Implementation**

**XII) Search Engine With Langchain Tools And Agents**

**A) Introduction To Tools And Agents**

**B) Creating Tools Using Langchain**

**C) Executing Tools And LLM with Agent Executors**

**D) End To End Search Engine GEN AI App using Tools And Agent With Open Source LLM**

**XIII) Gen AI Project-Chat With SQL DB With Langchain SQL Toolkit and Agentype**

**A) Demo of the Project**

**B) Preparing the Data For SQlite3 Database**

**C) Preparing The Data For My SQL Database**

**D) Creating the Streamlit Web app and Configuring the Databases**

**E) Integrating Web App With Langchain SQL Toolkit And Agenttype**

**XIV) Text Summarization With Langchain**

**A) Introduction To text summarization With Langchain**

**B) Stuff Chain And Map Reduce Text Summarization Indepth Intuition**

**C) Stuff And Map Reduce Summarization Impelmentation**

**D) Refine Chain Summarization Intuition And Implementation**

**XV) Gen AI Projects- Youtube Video And Website Url Content Summarization**

**A) End To End Project Demo**

**B) Implementing Youtube Video And Website Url Content Summarization GEN AI App**

**XVI) Text To Math Problem Solver Using Google Gemma 2**

**A) Demo of the End to End Project**

**B) End To End Text to Math Problem Solver Using Google Gemma2 Model Implementation**

**XVII) Huggingface And Langchain Integration**

**A) Introduction To Huggingface And Langchain Integration**

**B) Langchain And Huggingface Integration Practical Implementation**

**C) End to End Gen AI Project With Langchain And Huggingface**

**XVIII) Pdf Query RAG With Langchain And AstraDB**

**A) End To End Project With PDf Query RAG With Langchain And AstraDB**

**XIX) MultiLanguage Code Assistant Using CodeLama**

**A) End To End MultiLanguage Code Assistant Implementation**

**XX) Deployment Of Gen AI Apps In Streamlit and Huggingspace**

**A) Deployment OF Gen AI APP In Streamlit Cloud**

**B)  Deployment Of Gen AI App In Huggingface spaces**

**XXI) Generative AI with AWS (Bonus)**

**A) Life Cycle Of Gen AI Project In AWS Cloud**

**B) Introduction To AWS Bedrock With Implementation**

**C) Document Q&A RAG With Langchain And Bedrock**

**D) End To End Blog Generation Gen AI Using AWS Lambda And Bedrock**

**E) Deployment Of Huggingface OpenSource LLM Models In AWS Sagemakers With Endpoints**

**XXII) Getting Started With Nvidia NIM and Langchain**

**A) Building RAG Document Q&A With Nvidia NIM And Langchain**

**XXIII) Creating Multi AI Agents Using CrewAI For Real World Usecases**

**A) Youtube Videos To Blog Page Using CrewAI Agents**

**XXIV) Hybrid Search RAG With Vector Database And Langchain**

**A) Introduction To Hybrid Search**

**B) Reciprocal Rank Fusion In Hybrid Search**

**C) End To End Hybrid Search RAG With Pinecone db And Langchain**

**XXV) Introduction To Graph Databases And Cypher Query Language With Langchain**

**A) Introduction to Graph DB Wwith Langchain**

**B) What is Knowledge Graph**

**C) Creating Your Neo4j AuraDB Database Instance**

**D) RDBMS VS Graph Database**

**E) Neo4j Property Graph Data Model**

**F) Getting Started With Cypher Query Language**

**G) Intermediate To Advance Cypher Query Language**

**XXVI) Practical Implementation With Graphdb With Langchain**

**A) Getting Started-Creating Environment**

**B) Inserting Data In Graph DB With Python And Langchain**

**C) Creating GraphQuery Chain With Langchain**

**D) Prompting Statergies GraphDB With LLM**

**XXVII) Detailed Intuition and Implementation Of Finetuning LLM Models**

**A) What Is Quantization Indepth Intuition**

**B) LORA And QLORA Indepth Mathematical Intuition**

**C) Practrical Implementation fine Tuning Custom Data With Google Gemma Model**

**XXVIII) End To End finetuning LLM Models With Lamini Platform**

**A) End To End Finetuning LLM Models With Lamini AI Cloud**

**XXIX) Building Stateful, Multi-Actor Applications Using LangGraph**

**A) Introduction To Langgraph**

**B) Creating Chatbots Using Langgraph**

**C) Creating Chatbots With External Tools Workflow With Langraph**

**D) End to End Multi AI RAG Chatbots Using Langgraph And AstraDB**

## **Libraries**

1. **langchain_community.document_loaders - To import document loaders**

2. **langchain-text-splitters - Need this Library to import all Text Splitters**

**A) Langchain and OpenAI**

Got API Keys from OpenAI and Langchain

.env file is needed in VS Code to have "OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"

**Added python-dotenv in "requirements.txt" - To import all the things mentioned in .env file into Jupyter Notebooks**

**B) Data Ingestion with Document Loaders:**

1. Add the details on requirements on "requirements.txt" file - langchain, python-dotenv, ipykernel, langchain_community,pypdf

**2. Reading a Text File:**

**We can import everything from here** - from langchain_community.document_loaders import TextLoader; loader=TextLoader('speech.txt')

**Only if we write loader.load, it will load the text documents** text_documents=loader.load()

**3. Reading a Py PDF:**

from langchain_community.document_loaders import PyPDFLoader; loader=PyPDFLoader('attention.pdf'); docs=loader.load()

**type(docs) will be a list; type(docs[0]) - Will a document and inside we will have the Page Contents**

**4. Web Based Loader:**

from langchain_community.document_loaders import WebBaseLoader

import bs4

**We need to install "BeautifulSoup" when we do Scrapping Task**; **bs4.SoupStrainer** - Once we give this, we can give the class, which we can retrieve from the website

So, go ahead and add in requirements.txt (Beautiful Soip - **bs4**)

**We clicked on the Inspect Element in the Website and planned to take only specific portions from the Website (post-title, post-content, post-header)**

loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")
                     ))
                     )

  **Web Path in code - 1 URL; WebPaths - Many URLS**

  **5. Arxiv**

  **When we search for "Attention is all you need", it will take us to that specific website**

  Update requirements.txt with Arxiv

  1. from langchain_community.document_loaders import ArxivLoader; docs = ArxivLoader(query="1706.03762", load_max_docs=2).load()

**We are saying load maximum of 2 documents**

query="1706.03762", that is the research paper we are talking about in Arxiv

**6. Wikipedia Loader:**

from langchain_community.document_loaders import WikipediaLoader; docs = WikipediaLoader(query="Generative AI", load_max_docs=2).load()

**We searched for Generative AI and we automatically get 2 documents**

**C) Text Splitter Techniques - Recursive Character Text Splitter**

**Important Point: In Recursive Character Text Splitter, we have "["\n\n", "\n", " ", ""]"**

**In Character Text Splitter, we have only "\n\n"**

#### Text Splitting from Documents- RecursiveCharacter Text Splitters
This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

- How the text is split: by list of characters.
  
- How the chunk size is measured: by number of characters.

**Every LLM has its own limitation of Context Size; So we have to divide into Chunks**

**langchain-text-splitters - Need this Library to import all Text Splitters**

**1. Loading a PDF File** - from langchain_community.document_loaders import PyPDFLoader; loader=PyPDFLoader('attention.pdf'); docs=loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

**text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)**; We are saying, trying to split everything considering the chunk size of 500; 

**When diving text each chunk should have maximum of 500 Characters and then it can have overlap of 50 Characters**

final_documents=text_splitter.split_documents(docs)

**Error while Krish Working - He used create_documents; It can be used when working with Text files; here type(docs[0]) is already docs; So we can directly use split_documents**

**when we print print(final_documents[0]); print(final_documents[1]); There will be some repetition of words in 1, where it was at the end of 0; That is because we gave and allowed Overlap of 50 characters**

**2. Load Text file and convert it using create_documents**

## Text Loader

from langchain_community.document_loaders import TextLoader

loader=TextLoader('speech.txt')
docs=loader.load()
docs

speech=""
with open("speech.txt") as f:
    speech=f.read()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text=text_splitter.create_documents([speech])
print(text[0])
print(text[1])

**D) Text Splitting Technique - Character Text Splitter**

#### How to split by character-Character Text Splitter:

This is the simplest method. This splits based on a given character sequence, which defaults to "\n\n". Chunk length is measured by number of characters.

1. How the text is split: by single character separator.
  
2. How the chunk size is measured: by number of characters.

**1. Separating using Character Text Splitter:**

from langchain_text_splitters import CharacterTextSplitter

text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=100,chunk_overlap=20)

text_splitter.split_documents(docs)

**In Krish Video, It wasn't able to find separator; So gave error like "Created a Chunk Size of 470, which is longer than the specified 100"**

**2. Same way we try with "create_documents" (Like we did for Recursive Character Text Splitter)**

speech=""
with open("speech.txt") as f:
    speech=f.read()

text_splitter=CharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text=text_splitter.create_documents([speech])
print(text[0])
print(text[1])

### Which to use among - RecursiveCharacterTextSplitter and CharacterTextSplitter - Use RecursiveCharacterTextSplitter whenever you have a choice; It is the most generic one

**E) Text Splitting Technique - HTML Header Text Splitter**

**This will be handy when we work with HTML, URL**

from langchain_text_splitters import HTMLHeaderTextSplitter

1. **Copied HTML Tag from Internet**

**headers_to_split_on=[("h1","Header 1"),("h2","Header 2"),("h3","Header 3")]** - These are the Header Tags that we are using to Split

**In order to use these tags and do the splitting we will use** - html_splitter=HTMLHeaderTextSplitter(headers_to_split_on)

**html_header_splits=html_splitter.split_text(html_string)** - To split the text and save it in an variable

**Based on h1,h2,h3, we were able to get the Tags**

**All the Tags were created based on the headers we gave; Metadata information has also been shared based on Tags which we have splitted into**

**When we are splitting into Texts, we are able to get in the form of Documents**

2. URL Example:

url = "https://plato.stanford.edu/entries/goedel/"

**Same condition here to split** - headers_to_split_on = [("h1", "Header 1"),("h2", "Header 2"),("h3", "Header 3"),("h4", "Header 4"),]

**Similar codes to see the Splits too**

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url(url)
html_header_splits

**F) Recursive JSON Splitter**

We will learn on how to split from JSON data; Let's say we are getting a chunk of JSON data from an API, we will see how do we split that and pass it to LLM Models

**We can also use this with Recursive Text Splitter - If you need a hard cap on the chunk size consider composing this with a Recursive Text splitter on those chunks**

**Text is split on JSON Value; Chunk Size is determined by Number of Characters**

**1. Loading from API**

We will convert the API response into Chunks and see from it

import json; import requests

**json_data=requests.get("https://api.smith.langchain.com/openapi.json").json()**

**It is a nested JSON; JSON has a value, and inside the JSON there will be another values**

from langchain_text_splitters import RecursiveJsonSplitter; json_splitter=RecursiveJsonSplitter(max_chunk_size=300); json_chunks=json_splitter.split_json(json_data)

**Printing top 3 chunks to see the Output - for chunk in json_chunks[:3]: print(chunk)**

**2. Now it will be Key Value Pair; We will see how we can convert it to Documents**

## The splitter can also output documents

docs=json_splitter.create_documents(texts=[json_data])

for doc in docs[:3]: print(doc)

**3. To directly get the String Content:**

texts=json_splitter.split_text(json_data)

print(texts[0]); print(texts[1])

**G) Introduction to OpenAI Embedding**

**We will learn embedding (Converting to vectors) using OpenAI, Ollama, HuggingFace; There are not that only these three are there; Lot of other techniques are also there, but if these three are more than enough, with regards to accuracy** [OpenAI - Paid; Ollama and HuggingFace - OpenSource]

**Google Gemini has different embedding techniques; Claudy3 from Anthropic has different embedding technique**

**G) Introduction to OpenAI Embedding:**

**Embeddings - Converting Text to Vectors**

from dotenv import load_dotenv - Call the environment variable in the coding environment

We also require langchain-openai - Installed both in requirements.txt

**We are working with Langchain-OpenAI**

load_dotenv() - To load all the environment variables

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY") - Loading the envrionment variables

**We can check in OpenAI Website (Platform.openai.com) to check which and all models are there and choose the model for embeddings**

**Loaded this model from OpnAI**

from langchain_openai import OpenAIEmbeddings

**Testing the OpenAI Embedding to convert Text to Vectors**

text="This is a tutorial on OPENAI embedding"

query_result=embeddings.embed_query(text)

embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

**len(query_result) --> 3072; That is what mentioned in the Documentation too; It takes the sentence and converts it to 3072 Dimensions**

**We can also set our own dimensions using this --> embeddings_1024=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)**

text="This is a tutorial on OPENAI embedding"

query_result=embeddings_1024.embed_query(text)

len(query_result) - The length of this will be 1024

### Vector Embeddings and Storing in VectorStoreDB called ChromaDB

### Loading Data, Splitting it, Converting it to Vectors and Storing in Vector Store DB (ChromaDB)

**(i) Loading the Data**

from langchain_community.document_loaders import TextLoader; loader=TextLoader('speech.txt'); docs=loader.load(); docs

**(ii) Splitting Data into Chunks:**

from langchain_text_splitters import RecursiveCharacterTextSplitter; text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50); final_documents=text_splitter.split_documents(docs); final_documents

**(iii) Vector Embedding and Storing in VectorStoreDB:**

from langchain_community.vectorstores import Chroma

db=Chroma.from_documents(final_documents,embeddings_1024)

**(iv) Retrieve the Information from VectorStoreDB using Similarity Search**

query="It will be all the easier for us to conduct ourselves as belligerents"

retrieved_results=db.similarity_search(query)

print(retrieved_results)

**The text was able to search in the VectorStoreDB and we were able to get the output results**

**H) Ollama Embeddings**

**1. Download Ollama into Local from Ollama Website and Github**

**2. To use Models in Local**

**Once installation is done, open Command Prompt, "ollama run model_name(ex: gemma:2b)"**

**If already installed, we can go ahead and chat with it**

**3. Need langchain-community library**

**4.Coding Part:**

from langchain_community.embeddings import OllamaEmbeddings

embeddings=(
    OllamaEmbeddings(model="gemma:2b")  
)    (By Default it uses Llama2, if we didn't specify any model)


r1=embeddings.embed_documents(
    [
       "Alpha is the first letter of Greek alphabet",
       "Beta is the second letter of Greek alphabet", 
    ]
)

len(r1[0]) --> 2048, Gemma model creates vector of 2048 Dimensions

**embed_documents - List of texts to embed**

embeddings.embed_query("What is the second letter of Greek alphabet ")

**The above uses embed_query to embed only one sentence**

**If we compare vectors of above 2 codes, both will have similar vector values, as they are similar; First we said about Beta and later we asked what is the second letter of Greek Alphabet**

**We can also Pull Models from the Ollama Website**

### We can check for Other Embedding models from "https://ollama.com/blog/embedding-models"

**Trying with Other Model:**

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

text = "This is a test document."

query_result = embeddings.embed_query(text)

**len(query_result) - It uses 1024 vector dimensions**

**I) HuggingFace Embeddings**

**Langchain and HuggingFace have integrated; We can call any LLM Models**

**We need to create our API Key; Go to Settings --> Access Tokens; Create our Access Tokens**

**from dotenv import load_dotenv - To call that Token; Load all the Environment Variables**

**Use HF_TOKEN="Your_HuggingFace_Token**

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

**1. Need to install SentenceTransformers from HuggingFace - That Library is available for Embedding Techniques (Library we need is Torch)**

Hugging Face sentence-transformers is a Python framework for state-of-the-art sentence, text and image embeddings. One of the embedding models is used in the HuggingFaceEmbeddings class. We have also added an alias for SentenceTransformerEmbeddings for users who are more familiar with directly using that package.

**2. Also need to install langchain_huggingface to call embeddings and LLM Models (Langchain and Huggingface have integrated together)**

**3. Coding Part**

from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

**Embedding Text:**

text="this is atest documents"

**len(query_result) - 384 Dimensions**

**We can also embed documents, where we give multiple texts**

doc_result = embeddings.embed_documents([text, "This is not a test document."])

doc_result[0]

query_result=embeddings.embed_query(text)

**J) VectorStore - FAISS**

**FAISS - Facebook AI Similarity Search**

**First we will learn about FAISS, ChromaDB; Later we will use AstraDB in end-to-end projects**

**AstraDB specifically uses CassandraDB**

Facebook AI Similarity Search (Faiss) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.

**We need to get installed with faiss-cpu in VS Code**

**1. Coding Part:**

from langchain_community.document_loaders import TextLoader

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OllamaEmbeddings

from langchain_text_splitters import CharacterTextSplitter

loader=TextLoader("speech.txt")

documents=loader.load()

text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30)

docs=text_splitter.split_documents(documents)

**Storing the Vectors in the VectorStoreDB (FAISS)**

embeddings=OllamaEmbeddings()

db=FAISS.from_documents(docs,embeddings)

**Querying from Database**

**Querying** - query="How does the speaker describe the desired outcome of the war?"

**Doing Similarity Search** - docs=db.similarity_search(query)

**Getting only the first result** - docs[0].page_content

## Retrievers Example:

We can also convert the vectorstore into a Retriever class. This allows us to easily use it in other LangChain methods, which largely work with retrievers

Retriever is like a Interface, whenever we put a query it gets connected to the VectorStoreDB; Only when we convert this to Retriever; It lacks an interface which will able to retrieve the details from Vector Store and provide the response

**Reason is: Whenever we work with different LLM Models, we can't directly use VectorStoreDB, we need to use it as a Retriever and use it with LLM Models**

**Converting into Retriever**

retriever=db.as_retriever()

docs=retriever.invoke(query)

docs[0].page_content

### FAISS also uses Similarity Search, which uses L2 distance, Manhattan Distance

There are some FAISS specific methods. One of them is similarity_search_with_score, which allows you to return not only the documents but also the distance score of the query to them. The returned distance score is L2 distance. Therefore, a lower score is better.

**O) Introduction to Ollama and Setup**

**If we don't have credit balances and cost thing, we can use Ollama instead of OpenAI API Keys; We can use Ollama and OpenSource LLM's which we can run locally**

**Steps:**

1. Download Ollama from Website and Install; Once installed we can see the icon in our system

2. **To use and download the model** - Open Command Prompt and type "ollama run model_name"; If not there we can download it

3. Once downloaded, we can run it in our local machine and ask questions like "What is Generative AI" in local machine on Command Prompt; Write "exit" to exit the conversation with the Chatbot

docs_and_score=db.similarity_search_with_score(query)

**Passing Vectors instead of Sentences, or instead of any documents:**

embedding_vector=embeddings.embed_query(query)

docs_score=db.similarity_search_by_vector(embedding_vector)

docs_score - Here score is respect to with embedding vectors

**Saving VectorStoreDB to Local:**

db.save_local("faiss_index")

**To load the Saved VectoreDB**

new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

**We use allow_dangerous_deserialization, saying that we trust this particular file**

**Loading from saved VectoreStoreDB** - docs=new_db.similarity_search(query); docs

**K) VectorStore - ChromaDB**

Install ChromaDB --> langchain_chroma add in requirements.txt and pip install

**Building a sample VectorDB**
from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader

from langchain_community.embeddings import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

**Load Data:** - loader = TextLoader("speech.txt")
data = loader.load()

**Splitting:** - text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

**Embedding and Storing in a VectorStoreDB**

embedding=OllamaEmbeddings()

vectordb=Chroma.from_documents(documents=splits,embedding=embedding)

**Querying from VectoreDB:**

query = "What does the speaker believe is the main reason the United States should enter the war?"

docs = vectordb.similarity_search(query)

docs[0].page_content

**Saving to the Disk:**

vectordb=Chroma.from_documents(documents=splits,embedding=embedding,persist_directory="./chroma_db")

**Loading from the Disk:**

db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

docs=db2.similarity_search(query)

print(docs[0].page_content)

### ChromaDB also uses L2(Manhattan Distance); ChromaDB internally uses SQLite3 DB. Every vectors gets stored into the SQLite DB

**Saving as a Retriever and Invoking it**

retriever=vectordb.as_retriever()

retriever.invoke(query)[0].page_content

### Use this Link to explore all Vector Databases - https://python.langchain.com/v0.2/docs/integrations/vectorstores/

**L) Building Important Components of Langchain**

import os

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## Langsmith Tracking

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"]="true"

os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

**Need to install langchain-openai, to use the langchain and openai integration features**

**Also need to Install IpyKernel**

## Loading the Model

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-4o")

print(llm)

**Giving Input and Getting Response from LLM**

result=llm.invoke("What is generative AI?")

## The reason it takes time is that we will record all those with help of API using LangSmith

#### Once we go to "smith.langchain.com" everything will get tracked over there once we click on the project we created; It shows delay, latency, Tokens, Cost

### Chatprompt Template - We can create a Prompt Template - It is how we want the LLM to behave or to respond or what kind of role we want our LLM to do

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert AI Engineer. Provide me answers based on the questions"),
        ("user","{input}")
    ]

)

### Creating a Chain: chain=prompt|llm (Combining Prompt with the LLM) (Whenever we we a input, it goes to the prompt, then to the LLM, and we will get the response)

**It is like once the Input is given, how it flows**

response=chain.invoke({"input":"Can you tell me about Langsmith?"})

print(response)

### Everything will be tracked in LangSmith Portal

**type(response)** - AI Message (Means response is directly comming from a AI)

## Output Parsing: StrOutputParser (String Output Parsing)

**It is like getting the message from LLM and how we want to display it** (We can also create our own Custom Parser

from langchain_core.output_parsers import StrOutputParser

output_parser=StrOutputParser()

**chain=prompt|llm|output_parser** (Prompt to LLM to Output Parser)

response=chain.invoke({"input":"Can you tell me about Langsmith?"})

print(response)

**We will get the Output Directly and won't be having any Content over there (No particular Content Variable, we will get the message directly)**

### Everything will be tracked in LangSmith

**M) Creating Application using Langchain**

**Task - We have a website which has content, we need to extract text from it, we will take up the content or try to scrap it. After scrapping we will divide the entire content or entire text into Chunks and convert it to Vectors and use LLM with Prompt Engineering to get Output** 

[We require data ingestion technique (There is a technique in Langchain called Web Based Loader, PDF Loader]

## Need to Install Beautifulsoup4 for web scrapping because internally WebBasedLoader from LangChain uses BeautifulSoup4

**Also need to Install langchain_community**

**Coding Part:**

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## Data Ingestion--From the website we need to scrape the data

from langchain_community.document_loaders import WebBaseLoader

**Loading the Data:**

loader=WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")

docs=loader.load()

docs

## Load Data--> Docs-->Divide our Docuemnts into chunks dcouments-->text-->vectors-->Vector Embeddings--->Vector Store DB

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

documents=text_splitter.split_documents(docs)

## Embeddings:

from langchain_openai import OpenAIEmbeddings

embeddings=OpenAIEmbeddings()

## Storing in a VectorStoreDB

from langchain_community.vectorstores import FAISS

vectorstoredb=FAISS.from_documents(documents,embeddings)

## Query From a vector db

query="LangSmith has two usage limits: total traces and extended"

result=vectorstoredb.similarity_search(query)

result[0].page_content

**Created LLM to use for Retrievers and Chains**

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-4o")

**N) Understanding Retrievers and Chains**

### We will learn about Retreival Chain and Document Chain (And Combining with LLM's to get response)

### Querying from a VectorStoreDB will give us based on the context information that is available near those vectors.

### For asking much more meaningful questions, we need to provide context with respect to that particular question; So we will be using a Retrieval Chain

### We will use Retrievers in most RAG applications or Document QnA's

**Coding Part:**

from langchain.chains.combine_documents import create_stuff_documents_chain **Creating a chain for passing a list of documents to a Model**

### LLM's should also have a Context about any documents to answer more properly; So we use documents chain

### Creating a Custom Prompt - Answer the following question based only on the provided context; Context is the information provided to the LLM Model, regarding the information about documents or texts

### Context information means, Instead of searching on the whole page, we can go ahead and give that entire paragraph and search for that particular text in this particular paragraph; Through this search will also happen quickly

### How will we get that entire paragraph as a Context: That is where we will be using "create_stuff_documents_chain"

from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>


"""
)

document_chain=create_stuff_documents_chain(llm,prompt)

### Document Chain will be responsible in providing the prompt template the specific context information; Regarding any input we are specifically asking

### Document Chain - It is a Runnable Binder, Mapper, like Chat Prompt Template, Input variable is the Context which Human is giving with Prompt; Input Variable is the context information; We can also see ChatOpenAI happening in particular Memory Location; Along with ChatOpenAI we will also see String Output Parser

### Step by Step it will be: First is Chat Prompt Template, Then ChatOpenAI, Then String Output Parser

### All was combined together as a Chain; After going through the chain, it will be able to provide some context information, about the thing that we are searching; One by one it will go and give us the output based on the context 

### We will use that Document Chain and Invoke

from langchain_core.documents import Document

document_chain.invoke({
    "input":"LangSmith has two usage limits: total traces and extended",
    "context":[Document(page_content="LangSmith has two usage limits: total traces and extended traces. These correspond to the two metrics we've been tracking on our usage graph. ")]
})

### Input is what User is giving, and context is what we need to give; We imported the Document and gave the Context Manually by us

### If we are invoking it, based on context it should give us output; We gave context and saying it, from this search for this

#### However, we want the documents to first come from the retriever we just set up. That way, we can use the retriever to dynamically select the most relevant documents and pass those in for a given question.

### Retriever: VectorStoreDB has all the information stored in it; Retriever is a interface and is responsible if anybody asks any question or input then what happens is this interface will be way probably getting the data from VectorStoreDB

### We don't even have to do the Similarity Search; After we create the VectoreStoreDB, we will convert it to a Retriever

### Retriever is like a interface with respect to any input; We will pass the input through retriever and get the response from the VectorStoreDB (It is like a Pathway to get information from VectorStoreDB)

#### Converting it as a Retriever and a Retrieval Chain:

retriever=vectorstoredb.as_retriever() **(Converting VectorStoreDB to a Retriever)**

from langchain.chains import create_retrieval_chain **(Importing for creating a Retrieval Chain)**

retrieval_chain=create_retrieval_chain(retriever,document_chain) **(Creating a Retrieval Chain) - First Argument - Retriever, which will be taking information from the VectorStoreDB; Second Argument - Document Chain which will be responsible in giving the Context Information; Along with the Document Chain we will use the retriever**

### All step by step it will be created: Initially, it will have the runnable binding parameters, ChatPromptTemplate, ChatOpenAI, String Output Parser inside that we have Stuff Documents Chain to provide Context 

## Getting the Response from LLM by Invoking

response=retrieval_chain.invoke({"input":"LangSmith has two usage limits: total traces and extended"})

response['answer']

**Once we invoke it, it will invoke the entire answer**

### If we just print "response", it will have "Input", "Context", "Document", "Answer"; It will all be in the prompt template where we have created this Context

**P) GenAI Application using Ollama**

### from langchain_community.llms import Ollama

## Langsmith Tracking

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"]="true"

os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

**Steps:**

1. Installed Stremlit for the frontend

Need to go inside "cd 1.2-ollama" and then run "streamlit run app.py" in command prompt to see the Streamlit Frontend

2. from langchain_core.prompts import ChatPromptTemplate - Used to create our own Chat Prompt Template, which we want these Open Source Models to do, we can define using this

3. from langchain_core.output_parsers import StrOutputParser - Creating an StringOutput Parser

4. Defining our Prompt Template, which we want our AI Assitant to do:

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

5. Defining the Streamlit Framework:

st.title("Langchain Demo With Gemma Model")

input_text=st.text_input("What question you have in mind?")

6. Defining Ollama Gemma Model:

llm=Ollama(model="gemma:2b")

output_parser=StrOutputParser()

chain=prompt|llm|output_parser

7. If Input is given and pressed on Enter, take that particular chain and Invoke the chain by giving the question parameter; Later we wrote that response into our system

if input_text:
    st.write(chain.invoke({"question":input_text}))

###8. We are tracking it also in LangSmith

**Q) Tracking GenAI Application using LangSmith**

**Monitored through LangSmith**

Once we ask a question, Output has many key value pairs like, output: messages: context "Context given to model (You are a assistant to answer questions asked)", type, etc..

We can see Ollama, which uses Gemma model and see the human question asked and the response

We can see Latency, how much time taken, tokens, etc.. in LangSmith

**No cost involved because we are using a Complete Open Source Model from Ollama**

**R) Building Gen AI App using LCEL - Getting Started with Open Source Models using Groq API**

#### Groq is the Platform in which we will use the OpenSource Models

#### Groq is the AI Infrastructure Company that delivers fast AI Inference; We need inference the models that has been developed by Tech Giants; We need to really deploy these models somewhere

#### If we are using our inferencing or deploying, there will be a lot of charges

#### Groq LPUi Inference, LPU standing for Language Processing Unit is a hardware and software platform that delivers exceptional compute speed, quality, energy efficiency (Refer Documentations for Further)

Langchain Expression Language (LCEL) - Used to chain Components together

In this quickstart we'll show you how to build a simple LLM application with LangChain. This application will translate text from English into another language. This is a relatively simple LLM application - it's just a single LLM call plus some prompting. Still, this is a great way to get started with LangChain - a lot of features can be built with just some prompting and an LLM call!

We will learn about:

- Using language models

- Using PromptTemplates and OutputParsers

- Using LangChain Expression Language (LCEL) to chain components together

- Debugging and tracing your application using LangSmith

- Deploying your application with LangServe

**OpenAI is cost to use any models (GPT4 or anything)**

**Steps:**

1. Install Langchain

2. Go to Groq.com and sign in

**Groq has deployed several models from Top Giants in their cloud and to access them we need API Keys; They also use LPU Inferencing**

3. Click on GroqCloud and go to "API Keys"

4. Copy the API Key and paste it on .env file in VS Code

5. Code for:

import os

from dotenv import load_dotenv

load_dotenv() **Loads all the environment variables**

## Coding to Load both Open AI and Groq

import openai

openai.api_key=os.getenv("OPENAI_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

#### 6. Installing Langchain_Groq (!pip install langchain_groq) - This is responsible for interacting with any LLM Models deployed in Groq Platform
groq_api_key

#### 7. Using Groq Models

from langchain_groq import ChatGroq

model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

#### Langchain creates integration with every LLM Platform over there

**S) Building Gen AI App using LCEL - Building LLM, Prompt and StrOuput Chains with LCEL**

**Steps to do:**

1. Installing langchain_core

2. from langchain_core.messages import HumanMessage,SystemMessage

Whenevr we chat with LLM, we need to say to that, which is provided by Human Being and which is basically an instruction; SystemMessage is an instruction on how the LLM Model should work

3. Creating a Message

messages=[
    SystemMessage(content="Translate the following from English to French"),
    HumanMessage(content="Hello How are you?")
]

4. Invoking the specific message:

result=model.invoke(messages) 

5. Using the Output Parser:

**Along with the output we will also see, tokens used, completion time, prompt time, total time, etc...; We get it as AI Message, Means output from LLM Model**

#### We got ouput inside a "Content" in the AI Message; We want to get only the output message, so we will use "Output Parser"

from langchain_core.output_parsers import StrOutputParser

parser=StrOutputParser()

parser.invoke(result)

#### We have two components: Model.Invoke and StringOutput Parser

#### Using LCEL we can chain the components

chain=model|parser

chain.invoke(messages)

#### 6. Prompt Templates

**Whenever we invoke with messages, it goes to model and next we get String Output Parser**

#### We have one more efficient technique called "Prompt Templates"; Instead of givig list of messages, take a combination of user input and some application logic, where we will be able to give some instruction

#### The application logic will take raw user inputs and transforms into list of messages that can be passed to large models

from langchain_core.prompts import ChatPromptTemplate

generic_template="Trnaslate the following into {language}:"

prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)

result=prompt.invoke({"language":"French","text":"Hello"})

result.to_messages() **Converting to Messages**

## Chaining together components with LCEL

chain=prompt|model|parser

chain.invoke({"language":"French","text":"Hello"})

**T) Building Gen AI App using LCEL - Deploy LangServe Runnable and Chain as API**

**LangServe brought by LangChain and is integrated with Fast API**

**Steps:**

1. Install FastAPI, uvicorn, langserve, sse_starlette

2. We loaded our model, created our Prompt Template, Created chain and output parser

3. Run .py file from Command Prompt

#### 4. Go to the website and type "/docs" after the link

#### 5. Use the "POST/chain/invoke" to do the predictions

#### 6. We can also go to website "/chian/output_schema"

#### 7. We also do with the help of Postman. Copy "127.0.0.1:8000/chain/batch" in Postman "POST" request; We will get the same output in Postman too

# **VII) Building Chatbots With Conversation History Using Langchain**

**A) Building Chatbot With Message History Using Langchain**

So we are going to continue the discussion with respect to Lang Chain. And now I'm going to show you how we can go ahead and build a chat bot. Already in our previous video we have seen that how to build a simple LM application with LCL. And we have discussed about Lang Chain expression, uh, language where when we were able to, you know, uh, may, uh, combine the chains together, all the chain components and we were able to do the prediction. Right. So let's go ahead and let's start working with chat bots.

Now in this video we will go over an example of how to design and implement an LLM powered chat bot. This chat bot will be able to have a conversation and remember previous interaction. Note that this chatbot that we build will only use the language model to have a conversation. Okay, there are several other related concepts that you may be looking for, like conversational RAG, like how to enable a chat bot experience over an external source of data. We will probably discuss this in the upcoming videos. Then we will also be discussing about agents. We will be developing multiple projects. Okay, but in this video we will try to focus on LLM powered chatbots.

So first of all, as usual again over here I am going to use the open source uh models from Grok API instead of using OpenAI. Okay. So first of all what I will say I will go ahead and say "import os". Uh, along with this, I will go ahead and write "from dotenv import load_dotenv". And then we have this "load_dotenv()". Here we are just, uh, loading all the environment variables. All the environment variables okay. And then after this, uh, what do we exactly do? We basically get our "GROK_API_KEY", and you know, that we have in our environment variable which we had updated last time. So this is my grok API key. So I will just go ahead and call it right. And in order to call it I will just go ahead and write "os.getenv('GROK_API_KEY')". Perfect.

So if I go ahead and print my grok API key, this is the output. Uh, please do not use this. Anyhow, uh, after I upload all these videos, uh, in the course, uh, all keys will be deleted. Okay, so now I have my grok API key. Now the next thing is that we will go ahead and see how to call our LLM model. So that already we have seen over here. So here we will be importing "from langchain_grok import ChatGrok". And here we are going to use ChatGrok with model "gamma-2-9b". This is the most newest model that we are going to use over here. Let me just go ahead and execute this. And here I have actually given the Grok API key. Okay. So this is how you specifically load a model okay.

Um, now after loading a model, what I am actually going to do is that we are going to, first of all, start with basic things. Okay. So let me just go ahead and write "from langchain.core.messages import HumanMessage". Whenever I have a human message that basically whenever I'm using this particular class, uh, it indicates that a user is giving a message itself. Right. So let me just go ahead and write "model.invoke(HumanMessage(content='Hey hi my name is Chris and I am a chief AI engineer'))". So here you'll be able to see as soon as I give the message over here, this we are invoking with respect to this particular model, that basically means this information is going to my model "gamma-2" over here. And I'm getting the response "Hello Chris. It's nice to meet you as a chief engineer. What kind of projects are you working on these days? I'm always eager to learn about the exciting work being done in the field of AI." And these are my all the metadata info, like how much is the input tokens, what is the output tokens, and what is the total number of tokens.

So here, just a simple conversation by using a human message class, I've actually done it right. Still, I have not initialized anything from the system message itself. I have not told, uh, like, what kind of behavior the LM model should probably go ahead with right now. Let me do one thing. And, you know this, right? Whenever we get this AI message, that basically means we are getting the response from our model. Now let me go ahead and write "from langchain.core.messages import AIMessage". Now again here I'll go ahead and write "model.invoke([HumanMessage(content='Hi, my name is so and so and I'm a chief engineer'), AIMessage(content='Hello! In the field of AI...')])". Right. So this is the AI message that I've actually got a response. So now here I'm actually going to create a sequence of messages from this particular text. I got the AI, I'm just hard coding. Okay. I got this particular message from this particular model. And now I will go ahead and just trace again a human message. And now I will go and ask, "Hey, hey, hey, what's my name and what do I do?" Okay, so this is the question that I'm going to ask over here.

Now, this is amazing. Okay, let's see the output. What I'm actually going to do because I'm, I'm first of all feeding this particular information with respect to the human message "Hey, my name is Krish and I am a chief AI engineer", and from this AI message I got this entire value. So I am probably putting it up. And I'm asking it whether it is able to remember my name. It is whether it is able to remember the information that I have actually given since I'm passing it in the form of list of messages here. Right. I think it should be able to remember it.

Okay. So now once I go ahead and execute it, so here you can say "You said your name is Krish and you are a chief AI engineer, right? Is there anything else you would like me to remember about you?" Okay, so all this information is basically coming. That basically means, uh, one important thing is that whatever conversation I'm giving inside this list of messages with the human message, AI message, human message, it is also able to remember the previous context. Okay. Now that is where we are going to discuss about message history. And this message history will probably help you to understand all the context that it will help the LM to remember all the context that we are probably discussing about. This is just an idea that whether the LM model is able to remember things or not. But at the end of the day, whenever we develop real world application, there will be different, different sessions that will be happening. Right? Let's say I'm using ChatGPT. Someone is using ChatGPT, some other people are using some other LLM models and they are chatting. Right. How that particular models are able to remember their context, that is with respect to session. So in order to understand that how these sessions are managed, we will be discussing about a very important property which is called as message history. Okay. Now let's discuss about this message history. And this is a very important component altogether in the LangChain itself. Right. Which will actually help you to work with the LM model.

Okay. Now, what we really need to use or understand how message history works. So here I'm going to put a message saying that we can use message history class to wrap our model and make it stateful. We have to make it stateful so that it remembers all the context with respect to any kind of person who are actually interacting with that model. This will keep track of our inputs and output of the model and store them in some data store. Further interaction will then load these messages and pass them into the chain as part of the input. Okay, so let's see how this can be actually done. So first of all for this we require another library which is called as "langchain_community". So please make sure that you install this. And once you install this I have already done this particular installation so that it does not take much time. You have to go ahead and update in the "requirements.txt". Okay. So first of all go ahead and update this in the "requirements.txt" and make sure that whatever installation that you have done, you are doing it in your virtual environment. Okay. Perfect.

So from here, uh, what I will do is that I, uh, "LangChain_Community" is basically used because any integration related to message history will be available in this "langchain_community". Okay. So, uh, let me go ahead and import some very important libraries, let's say "from langchain_community.chat_message_histories import ChatMessageHistory". Okay. Inside this chat message history we will be importing "BaseChatMessageHistory". And then finally I will go ahead and import "from langchain.core.runnables import RunnableWithMessageHistory". Now this is something really important. Okay. Here you'll be able to see that we can import the relevant class over here, set up our chain which wraps the model and add in the message history. That is what we really need to do, right? Every message that we are probably putting up, it needs to be added in the message history itself, how it is done. I'll just talk about it sometime.

Okay. But the thing that you really need to understand is that whenever different, different users are chatting with the LM model, how we are going to make sure that one session is completely different from the other session. So for that reason, what we will do is that we will create one amazing function, okay. And this particular function will be definition "get_session_history". Now what this function is going to do, let me talk about it okay. Here this function will be creating a session ID, okay. This session ID and this session ID will be a string, okay. Right, I've written a session ID and this will be a string. Right. The return type of this particular function will be "BaseChatMessageHistory". So whatever chat history is basically getting created right that is imported from this particular library. So what I'm actually going to do is that this "get_session_history" with respect to a session ID will be created. Right. And this session ID will be used to distinguish one chat session with the other. Okay. So here, uh, the return type is "BaseChatMessageHistory". I will say "if session_id not in store", so let me just go ahead and create one variable called as "store" which will be a dictionary over here. So here I will just go ahead and write "store = {}". Okay. Let me talk more about it. And the explanation will be very good. And I will show you the practical example. I will go ahead and write "store[session_id] = ChatMessageHistory()". Okay. So now see this everything, please try to understand it right. First of all, for this particular message history, so you can see that we have a class which is called as "ChatMessageHistory". Okay. Now this chat message history, whenever we are creating a session ID right, we are storing the session ID in this particular dictionary. Right. With respect to this particular value, whatever session ID I'm actually creating, I am initializing a "ChatMessageHistory". Right. So that basically means this is an object of a chat message history. And whatever chat is specifically happening automatically, it should be going inside this session ID. Okay.

Next thing is that after this I will just go ahead and return "store[session_id]". Now see you will know there is some confusion with respect to the definition, but let's understand this definition. So what is the definition over here? This is a function which is called as "get_session_history". Whenever I give a session ID, it should be able to check whether the session ID is present in this particular dictionary or not. If it is present, we are going to get the entire chat message history from that and we will return it. Okay. And that is how we are going to distinguish it. Two things: this "get_session_history" will give you a response type of "BaseChatMessageHistory". It is an abstract class for storing the chat message history. If I talk about chat message history, it is an in-memory implementation of chat message history, stores messages in an in-memory list. So it is very much simple. Whenever I give a session ID it should first of all go ahead and check in this particular dictionary whether it is available or not. If it is available, it will go ahead and pick up the entire chat message that we have discussed with respect to, or whatever questions we have asked with respect to the LM model for that session ID, and it is just going to return it. Now, see how it is going to work. Okay. I'll execute this.

I will talk about this "RunnableWithMessageHistory" soon; it will be used as we go ahead. Okay. Now the best thing over here will be that we will be seeing how we will be using this. Okay. So first of all, we will go ahead and create our config. Okay. So I'll write "config = {'session_id': 'chat1'}". Now let's go ahead and execute this. Okay. Let's go ahead and execute. So this basically becomes my config, okay. My session ID is "chat1". Now let me use this particular session ID and let me chat with the LM model. So here you'll be able to see that I will write "response = RunnableWithMessageHistory(model, get_session_history).invoke(HumanMessage(content='Hi, my name is Krish and I am an AI engineer'), config=config)". Now see, I am trying to interact with based on the session ID. Okay, now see this all information is over here. But along with this we can go ahead and give our configuration. The configuration will be given inside this invoke. And here I will go ahead and write "config=config". Now when we are giving the config, that basically means for this session ID we are interacting. Okay. So based on the session ID it will be able to remember all this context.

So if I go ahead and execute this, okay. So here you can see "AIMessage(content='Hi Krish, it's great to meet you. Being a chief engineer is an exciting role.')". All this information is basically coming. Okay. Now if you want to get this particular content, I will also save this in my response variable. Okay. And let me just go ahead and write "response.content". Okay. I can also go ahead and write like this. So this is what is the message that I will be getting. Now let me do one thing, okay. Let me take another response or let me go ahead and again use this with message history. And this time I'm sending a message of human message like "What's my name?" And I'll write "config=config". Now in this case if I see the output here you'll it will say that "Hey, your name is Krish". Okay. White is able to remember it because I have given the same session ID.

So let's say that now if I change the config, will it be able to remember this particular context? Okay. Now here I'm just going to change the config. When I say change the config I'm changing the session ID, right. So I'm changing the session id. Okay. Now if I give the session ID some different session ID. So for giving the session ID I will again use this config over here. The first session id was "chat1". Now this will be my config one. Okay, let me just go ahead and give it. So this will basically be my config one. Okay. Now with respect to this particular config I will again go ahead and use my response and I'll say "RunnableWithMessageHistory.invoke(HumanMessage(content='What's my name?'), config=config1)". Now do you think it will be able to remember or not? Just try to guess, take a guess and make sure that you write down in the comment section the Q&A section that we have.

Okay, so here you'll be able to see that if I go ahead and execute "response.content", the reason it is able to remember because I have given the same session ID. Now let me change this particular session ID, I'll say "chat2". Okay. Now if I go ahead and execute it "RunnableWithMessageHistory.invoke(HumanMessage(content='What's my name?'), config={'session_id':'chat2'})", "Hey, as an AI, I have no memory of past conversation and don't know your name." Now what exactly I've actually done over here, I've just changed the session ID. Now, whatever conversation I have right now, it will entirely be saved in the session ID "chat2". Okay, so that in the later stages, whenever we use this with message history, it will just give the session ID of "chat2" and it will retrieve the information.

Now let me do one thing. Let me quickly go ahead and copy this entire thing. Okay. Let me go ahead and write and say "Hey, my name is John". Okay. Hey, my name is John. Okay. And now I will go ahead and execute with this config. Okay. So if I go ahead and say "Hey John, it's nice to meet you." Now if I go ahead and execute the same code with this particular config, okay. Now you'll be able to see that it will understand "Your name is John, I remember. Is there anything else I can help you with?" Okay, so with the help of this session ID, you'll be able to see how easily we are able to switch with respect to the context of conversation that we are having.

Okay. So I hope you got an idea. I hope you got an idea with respect to this particular message history, why we are specifically using it, but just an easy way to understand if just highlight it over that particular class, let's say over here, "BaseChatMessageHistory" is there. This is nothing, but this is an abstract base class for storing chat message history and chat. "ChatMessageHistory" is basically present over here. This is an in-memory implementation of chat message history. As soon as I give my session ID whenever I call this. So what is "with_message_history"? It is a runnable. It is a chain between model and "get_session_history". Whenever I am going to interact with the model, it is first of all going to, uh, give "get_session_history". That basically means based on the session ID, it is going to pick up your entire chat session. Okay. And based on that it is going to remember the context.

So this was just an idea with respect to implementing, uh, how you can basically work with chat history whenever you're developing a chat bot. Now we will try to implement all this chat history, uh, in our prompt template. And that is what we are going to see in the next video. So here I hope you are able to understand this. This was it from my side. I'll see you all in the next video where I will be discussing about Prompt template. So thank you. I'll see you all in the next video.

**B) Working With Prompt Termplate And Message Chat History Using Langchain**

So we are going to continue the discussion with respect to creating this chat bot.

So already we have seen about the message history and how we can remember the context.

Now let's go ahead and use prompt templates. Prompt templates help to turn raw user information into a format that LLM can work with in case the raw user input is just a message, which we are passing to the LM.

Let's now make that a bit more complicated. First, let's add in the system message and some custom instructions. Till now we were just passing a list of messages, right? Now let's go ahead and work with this particular prompt template.

For working with prompt templates, I will write "from langchain.core.prompts import ChatPromptTemplate" to import the chat prompt template class.

Along with this, I will also use a message placeholder. Previously, we used to give the messages directly, like human message and AI message in a list. But here, we will just go ahead and use a message placeholder.

Now let's quickly go ahead and create my prompt. I will write:

"chat_prompt_template = ChatPromptTemplate.from_messages([{'role': 'system', 'content': 'Hey, you are a helpful assistant. Answer all questions to the best of your ability.'}, {'role': 'input', 'variable_name': 'messages'}])"

Here, we are providing system information first. The system role instructs the model to be a helpful assistant and answer questions to the best of its ability. The input placeholder is called "messages", which will be used to pass human messages dynamically.

Once we have the prompt template, we can create our chain:

"chain = chat_prompt_template"

Whenever we invoke this chain, whatever human message we give needs to be provided as a key-value pair where the key is "messages". Previously, we didn’t use a message placeholder, and messages were passed as a list. Here, we are using a message placeholder, so any human input will go into the "messages" variable.

To invoke the chain, we can write:

"response = chain.invoke({'messages': {'content': 'Hi, my name is Krish'}})"

This will pass the human message through the chain. The chain will then process it, taking the system instructions into account, and return a response.

Now, if we want to add multiple input variables, we can extend this. For example, we can have a language variable:

"chat_prompt_template = ChatPromptTemplate.from_messages([{'role': 'system', 'content': 'You are a helpful assistant. Answer all questions in the language provided.'}, {'role': 'input', 'variable_name': 'messages'}, {'role': 'input', 'variable_name': 'language'}])"

Then, when invoking the chain, we pass both keys:

"response = chain.invoke({'messages': {'content': 'Hi, my name is Krish'}, 'language': 'Hindi'})"

This will allow the assistant to respond in Hindi.

Next, to manage chat history, we can wrap this chain in a message history class. We need to specify the input key used to save chat history. For example:

"with_message_history = RunnableWithMessageHistory(chain=chain, get_session_history=get_session_history, input_key='messages')"

Then, invoking this with both keys looks like:

"response = with_message_history.invoke({'messages': [{'content': 'Hi, I am Krish'}], 'language': 'Hindi'})"

Finally, we can display the response:

"print(response.content)"

This way, the model will understand the human message, the language, and the chat history context.

Managing conversation history is very important because the history can become long, and LLMs have context window limitations. In the next video, we will see how to manage the conversation history efficiently.

This was it for my side. I will see you all in the next video. Thank you. Take care.

**C) Managing the Chat Conversation History Using Langchain**

So we are going to continue the discussion with respect to creating the chatbot. Already in our previous video, we discussed prompt templates and conversation history. Now, we are going to focus on understanding how to manage conversation history.

One important concept when building a chatbot is that if conversation history is left unmanaged, the list of messages will grow unbounded and potentially overflow the context window of the LM. Therefore, it is important to add a step that limits the number of messages you pass.

To do this, we can import the following:

"from langchain.core.messages import SystemMessage, TrimMessages"

Here, "SystemMessage" helps define system instructions, while "TrimMessages" is a helper to reduce the number of messages being sent to the model. The trimmer allows specifying how many tokens to keep, whether to always retain system messages, and whether to allow partial messages.

For example, we can initialize the trimmer like this:

"trimmer = TrimMessages(max_tokens=70, strategy='last', token_counter=my_model_token_counter, include_system=True, partial=False)"

Here, "max_tokens" limits the number of tokens, "strategy='last'" ensures the last messages are prioritized, "token_counter" is a function or model for counting tokens, "include_system=True" retains system instructions, and "partial=False" ensures only complete messages are included.

Next, let's define a set of messages:

"messages = [
SystemMessage(content='You are a good assistant.'),
{'role': 'human', 'content': 'Hi, I am Bob.'},
{'role': 'ai', 'content': 'Hi!'},
{'role': 'human', 'content': 'I like vanilla ice cream.'},
{'role': 'ai', 'content': 'Nice!'},
{'role': 'human', 'content': 'What is 2 plus 2?'},
{'role': 'ai', 'content': 'Four.'},
{'role': 'human', 'content': 'Thanks.'}
]"

Now, if we apply the trimmer:

"trimmed_messages = trimmer.invoke(messages)"

By setting "max_tokens=45", older messages like "Hi, I am Bob" may be trimmed off, keeping only the most recent conversation within the token limit. This helps the LM model focus on relevant context while respecting its context window.

We can also integrate this trimmer in a chain. First, import the required libraries:

"from operator import itemgetter"
"from langchain.core.runnables import RunnablePassthrough"

We can then create a chain with the trimmer applied to the messages:

"chain = RunnablePassthrough().assign(messages=itemgetter('messages')).map(trimmer).concat(prompt).run(model)"

To invoke this chain with new human input, we pass messages as a key-value pair and other variables like language:

"response = chain.invoke({
'messages': [{'role': 'human', 'content': 'What ice cream do I like?'}],
'language': 'English'
})"

"print(response.content)"

You may notice that older context like "I like vanilla ice cream" can be trimmed due to the token limit, which is why the model may respond without knowing your favorite ice cream. However, simpler questions like "What is 2 plus 2?" will still be answered if the context is within the retained tokens.

Finally, we can wrap this entire setup in a message history class:

"with_message_history = RunnableWithMessageHistory(chain=chain, get_session_history=get_session_history, input_key='messages', session_id='chat5')"

Now, when invoking this, the model keeps track of conversation history in a managed way. For example:

"response = with_message_history.invoke({
'messages': [{'role': 'human', 'content': 'What is my name?'}],
'language': 'English'
})"

"print(response.content)"

This setup ensures conversation history is managed, the trimmer limits context size, and messages are passed in key-value pairs to the LM.

Overall, we covered managing messages, applying trimmers, integrating with chains, and using message history classes. These are critical components when building a functional chatbot that can handle long-term interactions efficiently.

This was it from my side. I hope you liked this video. See you in the next video. Thank you.

**D) Working With VectorStore And Retriever**

So we are going to continue the discussion with respect to Lang Chain.

Already, we had, in our previous module, discussed about building chatbots with chat message history.

Now in this video, we are going to discuss about vector stores and retrievers.

In this video tutorial, we will familiarize you with the LangChain vector store and retriever abstraction. These abstractions are designed to support retrieval of data from vector databases and other sources for integration with LLM workflows. They are important for applications that fetch data to be reasoned over as part of model inference.

If you remember, previously we have discussed vector stores and retrieval, but here we really want to add more features and even play with chat message history.

First of all, we will quickly go ahead and install the important libraries like pip install langchain, along with pip install langchain[chroma]. I hope we have done all this installation beforehand in our same virtual environment.

Along with this, you should also have done the installation of pip install langchain/grok. Make sure you write all these in your requirements.txt.

Now, the next thing that we are going to do is take an example given in the LangChain documentation page of how to create a document. You need to understand what exactly this document is.

Let's define some information about the document:

# Page content and metadata
from langchain.docstore.document import Document

documents = [
    Document(page_content="This is the content of document 1", metadata={"source": "doc1"}),
    Document(page_content="This is the content of document 2", metadata={"source": "doc2"}),
    Document(page_content="This is the content of document 3", metadata={"source": "doc3"})
]


LangChain implements a document abstraction which represents a unit of text and associated metadata. It has two attributes: page_content, a string representing the content, and metadata, a dictionary containing arbitrary metadata. The metadata can capture information about the source of the document, its relationship to other documents, and other details.

An individual document object often represents a chunk of a larger document. For example, a PDF with eight pages can be represented as eight documents, one per page.

Now let's go ahead and work with Vector Store.

The main purpose of a Vector Store is to convert text into embeddings, or word vectors, and store these vectors in a database. In this example, we will use Chroma:

from langchain.vectorstores import Chroma


We will use open-source libraries and models for embeddings. Make sure your Hugging Face token is set in your environment:

import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HDF_TOKEN")
os.environ["HDF_TOKEN"] = hf_token


Now, let's import the LM model from Grok and initialize it:

from langchain.grok import Grok

lm_model = Grok(
    api_key=os.getenv("GROK_API_KEY"),
    model="llama-3-8b-8192"
)


To use Chroma, we need embeddings. We can use Hugging Face embeddings:

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


Now we can convert documents into vectors and store them in Chroma:

vector_store = Chroma.from_documents(documents, embedding=embeddings)


This converts each document into a vector using the embedding model and stores it in Chroma DB. We can now perform similarity searches:

results = vector_store.similarity_search("Cat")


You can also get similarity scores:

results_with_scores = vector_store.similarity_search_with_score("Cat")


Now let's discuss retrievers.

Vector stores cannot be directly integrated into LangChain chains using LCL because they are not subclasses of Runnable. However, retrievers are Runnable and can be incorporated into LCL chains.

You can create a simple retriever using RunnableLambda:

from langchain.core.runnables import RunnableLambda

retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
retriever.batch(["cat", "dog"])


This executes a similarity search for each input in the batch.

Another method is to use the vector store's built-in retriever:

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
retriever.batch(["cat", "dog"])


This converts the vector store into a retriever interface that can be queried easily.

Finally, we can integrate the retriever with a chain:

from langchain.prompts import ChatPromptTemplate
from langchain.core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

rag_chain = SomeRAGChain(
    question=RunnablePassthrough(),
    context=retriever,
    prompt=prompt,
    llm=lm_model
)

response = rag_chain.invoke({"question": "Tell me about dogs"})
print(response.content)


This is a basic RAG (Retrieval-Augmented Generation) implementation:

chat_prompt_template defines the prompt.

runnable_pass_through passes the question input.

retriever supplies the context from the vector store.

The LLM generates a response based on this context.

With this setup, you can retrieve relevant information from your documents and use it to answer questions in a chatbot workflow.

This covers the basics of vector stores, retrievers, and RAG integration with LangChain.

# **VIII) Conversational Q&A Chatbot With Message History**

# **A) Building Conversational Q&A Chatbot With Message History**

So we are going to continue the discussion with respect to Lang Chain. And now in this particular video we are going to talk about conversational Q&A chatbot. So we will be building some amazing conversation Q&A chatbot. You will be understanding how you can go ahead and build this.

Now see this in many Q&A application. We want to allow the users to have a back and forth conversation, meaning the application needs some sort of memory of past question and answers, and some logic for incorporating those into its current thinking. In this guide, we focus on adding logic for incorporating historical messages. Further details on the chat history management is covered in the previous videos, right? We have already discussed about some of the chat history management terms and now we will be covering two approaches. One is in chains in which we will always execute a retrieval step, and the second one is agents, in which we give an LLM discretion over whether or how to execute a retrieval step for multiple steps. So already we have discussed so much things about chat history. Right. But the main idea of using this chat history specifically for a Q and a chat bot. So in this video I will go ahead and explain you that okay.

Now first of all, uh, over here also, I'll be making sure that I use open source models. So for this I will be again importing "import OS" and then from here I will go ahead and uh write "from dot env import load_dot_env". Right. So we are also going to go ahead and import this. Let's see whether it is executing fine. So we will go ahead and initialize this "load_dot_env()" because we will be having all our environment variables. Now with respect to the environment variables, the first key that I will be requiring. And here I will go ahead and write "from langchain_grok import ChatGrok" okay. And I think we have repeated this code many multiple times. Right. So I'm just showing you again and again so that you don't forget this okay. Now when I'm importing this ChatGrok along with this you'll be able to see that I will also go ahead and take my API key from my environment variable. So for this I will go ahead and write "OS.get_env('grok_api_key')" okay. So once we get this uh the next step will be that we will go ahead and call our LM model specifically over here. So for this we will go ahead and use llama three in this particular use case okay. So this is where we specifically get our LM model. So this is the fundamental thing that we should definitely do in each and every um, basically whenever you are developing most of the application, this is common. If you are really interested to just work with open source model. Grok is the grok. Infrastructure is the thing that you should really go ahead with.

Okay, now let's work on some of the important things. So first of all, what I will do is that I will go ahead and install "pip install bs4". Okay. So this is a library which is called as Beautifulsoup for this is used to extract a HTML page okay. Extract the content of the HTML page. So here you can see that the requirement is already satisfied. I have selected this venv environment and make sure that whenever you do this installation, you update that in the "requirements.txt" in this specific project.

Okay, now the first thing what I am actually going to do over here is that I will quickly go ahead and upload or import some of the libraries. Okay, so I'm going to import something called as chroma. So here in this code I am going to go ahead and import "chroma" from "lang_chroma". Along with this I am going to use a document loaders and will be importing from a web based loader. Because I want to read the content of a web page and we'll try to create a conversational Q&A chat bot. So that basically means I'll be having a website which will be my external data source. And from that I will try to chat through my chat bot. Okay. And whatever conversation we basically add, we will also add our message history. Right then. Along with this we are going to use this chat prompt template and you know where it is available inside prompts. Now instead of using OpenAI embeddings, I will remove this okay, because I will not be using OpenAI embeddings because it will charge you some amount of money. Instead, what I will do, I will go ahead and use this particular code wherein I'll go ahead and import my "HF_token". If you remember, how did I import h_token along with this in my environment variable, this is also present okay. So "from langchain_huggingface import HuggingfaceEmbeddings" and "HF_token" from where do you get it is you get it from your huggingface account. Right. So this is the embedding that I'm actually going to use, right? Huggingface embedding. So let's go ahead and execute this. Perfect. Till here I think everybody everything looks fine. And then after this we are going to also import recursive character text splitter so that we will be able to split our document and chunks okay. So once we upload or once we load all these things now we are also going to use two important functions also. Right. And this functions will discuss about it. So one is "from langchain.chains import create_retrieval_chain". Okay. And here whenever we try to create a chain by using this library or by using this function we will be able to create chains okay. Create retrieval chain like let's say if I if you know about retrieval, retrieval is nothing but it is an interface to a vector store database. Right. And if I want to create a chain with this particular retriever, I have to use this particular function which is called as "create_retrieval_chain". Okay. We'll discuss more about this as we go ahead. The next thing that we are going to use is something called as combined documents create stuff document chain okay. So as you go ahead, uh, in the future, right. In an upcoming, uh, sessions and modules, I will be talking about text summarization. Now when we discuss about text summarization, there we will be discussing about three types of text summarization. One is stuff, one stuff document chain, one is map reduce and one is refine in stuff document chain. What it does is that it combines all the documents and then it sends it to the prompt template. And that is why we will be using this specific function. Okay. So I will show you everything with an example.

Okay. Now let us go ahead and execute this okay. So here we have got executed it. Now I'm going to use a web based loader. And let's say I'm going to use this particular website okay. So let me just go ahead and quickly import BS4 also. So I will go ahead and write "import BS4" because I'm going to use BS Beautifulsoup. For now here inside this web loader I'm giving my web path. This is the URL that I'm trying to read the content from okay. And then here you'll be able to see I'm using a class which is called as post content from that page and post title. So if I go ahead and hit this page, let me just show you by opening this page over here, okay. Let me open my browser and let me just hit this page. Okay. So here you can see that this is the entire page content. And I will be using this particular page content as my external data source. Okay. Now from this what all fields I'm really interested in. So I am interested in getting the post content field, post title field and post header field. So if I go back if I just go ahead and do the inspect. So here you'll be able to see I'm getting the post title field, post content field and post header. Post header will also be somewhere here only. Yeah here. So all this information will try to extract from this. Now, uh, let me close this console. Let me go back to my code.

Okay. Now, once I probably, uh, use this loader that is a web based loader. This is for the data ingestion from a website. I will go ahead and write "loader.load()" and I will load all the documents. Okay. So here I will be able to get my documents okay. So if I go ahead and execute this and see my documents you'll be able to see that one single document I'm able to get over here. And all the information is basically shown over here inside this particular page content. Perfect. Now usually what we do is that whenever once we get all this content, it is a good practice that we try to divide this entire document into chunks of document. The reason is very simple, because every LM models specifically have some kind of context size, right. So it is better that we try to break all this particular documents into chunks. So for this we will be creating a text splitter variable. And here we will be using some recursive character text splitter. And let me go ahead and write. Our chunk size is equal to 1000. And here we are going to basically use chunk_overlap with 200 okay. So once we actually do this uh then we are basically going to create our text splitter over here. The next step, uh, what we are going to do is that I'm just going to write "splits = text_splitter.split_documents(docs)" and whatever docs I have I will be giving over here. So it will basically go to do the splitting. And after doing the splitting I will go ahead and store this entirely in our vector store database. Right? So for that I will write "from chroma import from_documents" and I will give "documents=splits" okay. And here I'm going to apply my embedding which will be equal to this particular embeddings that I have actually used. That is nothing but huggingface embedding. Right. So here you can see that I have initialized this Huggingface embedding. Perfect. So till here everything looks good. Uh, this will finally give me my vector store okay. Vector store. And this we have already reported in our previous video. Right. And if I really want to convert this into a retriever, all I have to write is that I have to write "vector_store.as_retriever()". Right. Here is your entire retriever right now. So this is the retriever, uh, over here. Okay, perfect. Let's see. So here is your entire retriever that we are able to get it. Okay. Now, this is fine. Now it's time that, uh, once we get this entire retriever. And this is for the vector store retriever basically means it is an interface with respect to vector store. So that whenever I ask any query, I should be able to get it from the vector store itself. Now this is important. And this you have actually seen it. Okay.

Now it's time. We will go ahead and define our prompt template okay. See the same thing what we are doing. Because after this I'm going to execute more things over here okay. So let me do one thing. Let me go ahead and define my system prompt. So this is my system prompt. I'm saying "Hey you are an assistant for question answer task. Use the following piece of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentence maximum. Keep the answer concise." Okay. And here a new line. And here we will be passing the context. Now what this context will be getting replaced with I will talk about it okay. Then I'm going to take the system prompt. I'm going to use it as a chat prompt template. "ChatPromptTemplate.from_messages(system_prompt, human_message=human_input)". So here we can actually go ahead and pass this particular input. And this is my system prompt, which is basically saying how the system should behave or how the LM model should act like, okay. Now this is absolutely fine.

Now my time, like now is the time that we need to attach this retriever along with the system prompt and along with the prompt. Right. So here we really need to create this particular chain because this retriever is pointing to the vector store database. This is my system prompt based on which I should be able to get my information. Um, which is this is specifically my prompt. So I need to probably go ahead and create a chain. Now the one thing is that how do I get this context information? Now to get this context information I will first of all go ahead and create my question answer chain okay. And here we will go ahead and write "qa_chain = create_stuff_document_chain(LLM=llm, prompt=prompt)". And here we are going to pass LM comma prompt. Now the reason I'm passing my LM model and prompt okay what this exactly is going to do okay. And that is the reason we have imported this create_stuff_document_chain I've already told that this create stuff document chain is going to make sure that it will combine all the document and it will pass in the form of the specific context in that specific chain. When we create the chain with respect to system prompt with respect to this particular prompt, LM model. Right. You'll be seeing that when we use Create stuff document chain. Here I'm passing two parameters LM and prompt. Now this prompt right is requiring some context information. Right. And this context information needs to be provided with respect to all the list of documents. Like what list of documents you have over here. Right. All this list of documents. Right. We did this particular split and all right. So that it will combine and it will send it to this particular prompt itself in the form of context. Right now, if we are doing this, the next step will be that we need to go ahead and create our entire retrieval chain. For creating the retrieval chain, we will be using a "create_retrieval_chain()". Now inside this create retrieval chain I'm going to make sure that I pass my retrieval along with this I also pass my question answer chain right. So this create stuff document chain will also get passed over here. Now I have retriever I have LM model I have prompt. So these all are connected in a case of a retrieval chain itself where I will be able to get the response okay. Now once I execute this, now it's time that we will go ahead and execute it. So in order to execute it I'll just write "response = retrieval_chain.invoke(input='What is self reflection?')" and here we are going to give my input. So I know my key is basically a input over here I will write "input='What is self reflection?'" okay. I'll ask this question. Okay. From here. So here, if I go ahead and ask what is this? I should be able to get my response. And my response is nothing, but it will be over here. And with respect to this, I will go ahead and write "print(response['answer'])". Perfect. And here you can see self reflection is a mechanism that allows autonomous agents to improve iteratively by refining action and more information. Okay, now here you can see we have used built in chain constructors, which is called as create_stuff_document_chain and create_retrieval_chain. Right. But at the end of the day, the basic ingredients that you will be able to find is something like a retrieval, prompt, LLMs. Right. Just a way to how to replace data with respect to this particular context and how to replace, um, how to probably get the answer with the help of this particular retrieval.

Right. But now it's time that we start working towards adding chat history. Okay. So here we are going to basically go ahead and add our chat history. Now how to add this chat history right. Let's say I've asked a question over here right. What is self reflection. Now I will again go to go to this and ask another question over here. Let's say I'll say I'll just continue and ask. This is my invoke. And here I'm just going to take the input. Now instead of writing "What is self reflection?" I'll say "How do we achieve it?" Okay. So this is what I'm going to ask. Now here you can see achieving task decomposition. It is basically same okay I have I was trying to ask with respect to what is self reflection. But here it is probably giving me some other thing like how do we achieve it. Now I wanted the information with respect to self-reflection. Now, the problem with this rank chain is that it's not able to understand the context, right? It is because it obviously does not have any kind of chat history. So what I will do now is that I will go ahead and show you how we can go ahead and add chat history, along with the prompt template, and how we can do the same task and see how things work with chat history over here.

Okay. So let's go ahead. And first of all, what I'm actually going to do is that for adding chat history, I will be importing "from langchain.chains import create_history_aware_retriever". Now the kind of retriever that I am going to create or the chain of retriever that I'm actually going to create. See here also we created a chain of retriever only. Right. So this is nothing but "create_retrieval_chain". Right. So this is nothing but a retrieval chain. But when we use this "create_history_aware_retriever", that basically means the retriever will also know about the history of information or history of conversation that has been basically made with the LM model. Okay. So we are going to probably create our chain with respect to with the help of this particular function. Okay.

The next thing is that I will go ahead and use "from langchain.prompts import MessagesPlaceholder". And why I have to use this because I will be able to create a chat history. Uh. Uh, I'll just say that, okay. This can be a key value pairs, where I will be storing all the information. Right. And that is where I will be using this MessagesPlaceholder. Just wait for a second. I will just make sure that in the coding will be understanding where we use this MessagesPlaceholder. Right. If we define a MessagesPlaceholder with any variable in that particular variable, only all the chat history will be saved. Okay, this is what I really want to talk about.

Okay, now I will go ahead and create a prompt okay. So this is basically called as continuous contextualized Q system prompt okay. So I'll say "Given a chat history and the latest user question which might reference the question in the chat history, formulate the standalone question which can be understood without the chat history. Do not answer the question, just reformulate it if needed. Otherwise return it as it is." Okay, so this is the kind of prompt that I have actually created over here. I'm saying, hey, please, you have the chat history. You have the later user question which might reference the context in the chat history, formulate a standalone question which can be understood without the chat. Do not answer the question, just reformulate if needed. Otherwise return. That is okay. So simple prompt that we are going to use over here.

Now for this we will be using uh I will be creating a prompt template. Now see I have given the system message with respect to this. And this is where I'm going to use MessagesPlaceholder. Now in this MessagesPlaceholder I'm creating a variable like "chat_history" where all my history of conversation will be stored. Okay. And here is my human input okay. So guys now here you can see that I have created my chat prompt template.

Now if I just want the answer over here I will go ahead and write "response['answer']". Perfect. And here you can see self reflection is a mechanism that allows autonomous agents to improve iteratively by refining action and more information. Okay, now here you can see we have used built-in chain constructors, which is called as "create_stuff_document_chain" and "create_retrieval_chain". Right. But at the end of the day, the basic ingredients that you will be able to find is something like a retrieval, prompt, LLMs. Right. Just a way to how to replace data with respect to this particular context and how to probably get the answer with the help of this particular retrieval.

But now it's time that we start working towards adding chat history. Okay. So here we are going to basically go ahead and add our chat history. Now how to add this chat history, right. Let's say I've asked a question over here, right. What is self reflection. Now I will again go to this and ask another question over here. Let's say I'll say I'll just continue and ask. This is my invoke. And here I'm just going to take the input. Now instead of writing "what is self reflection" I'll say "how do we achieve it". Okay. So this is what I'm going to ask. Now here you can see achieving task decomposition. It is basically same. Okay I have I was trying to ask with respect to what is self reflection. But here it is probably giving me some other thing like how do we achieve it. Achieving task decomposition is basically selecting. It is a tree of thoughts involves so and so. But I wanted the information with respect to self-reflection. Now, the problem with this rank chain is that it's not able to understand the context, right? It is because it obviously does not have any kind of chat history. So what I will do now is that I will go ahead and show you how we can go ahead and add chat history, along with the prompt template, and how we can do the same task and see how things work with chat history over here. Okay.

So let's go ahead. And first of all, what I'm actually going to do is that for adding chat history, I will be importing "from langchain.chains import create_history_aware_retriever". Now the kind of retriever that I am going to create or the chain of retriever that I'm actually going to create. See here also we created a chain of retriever only. Right. So this is nothing but "create_retrieval_chain". Right. So this is nothing but a retrieval chain. But when we use this "create_history_aware_retriever", that basically means the retriever will also know about the history of information or history of conversation that has been basically made with the LM model. Okay. So we are going to probably create our chain with respect to with the help of this particular function. Okay. The next thing is that I will go ahead and use "from langchain.core.prompts import MessagesPlaceholder". And why I have to use this? Because I will be able to create a chat history. Uh. Uh, I'll just say that, okay. This can be a key-value pairs, where I will be storing all the information. Right. And that is where I will be using this "MessagesPlaceholder". Just wait for a second. I will just make sure that in the coding we will be understanding where we use this "MessagesPlaceholder". Right. If we define a "MessagesPlaceholder" with any variable, in that particular variable only all the chat history will be saved. Okay, this is what I really want to talk about.

Okay, now I will go ahead and create a prompt. Okay. So this is basically called as continuous contextualized Q-system prompt. Okay. So I'll say "given a chat history and the latest user question which might reference the question in the chat history, formulate the standalone question which can be understood without the chat history. Do not answer the question, just reformulate it if needed. Otherwise return it as it is." Okay, so this is the kind of prompt that I have actually created over here. I'm saying, hey, please, you have the chat history. You have the latest user question which might reference the context in the chat history, formulate a standalone question which can be understood without the chat. Do not answer the question, just reformulate if needed. Otherwise return. That is okay. So simple prompt that we are going to use over here.

Now for this we will be using, I will be creating a prompt template. Now see I have given the system message with respect to this. And this is where I'm going to use "MessagesPlaceholder". Now in this "MessagesPlaceholder" I'm creating a variable like "chat_history" where all my history of conversation will be stored. Okay. And here is my human input. Okay. So guys now here you can see that I have created my chat prompt template. Now instead of using a retriever I will go ahead and create this history-aware retriever. Right now I will get a new type of retriever instead of the previous retriever. And here I will write "history_aware_retriever = create_history_aware_retriever(retriever=retriever, contextual_cue_prompt=Q_prompt)". In short, what we are doing over here is that we are upgrading this retriever to history-aware retriever, which will also be able to retrieve the results, or get the results from the vector store DB, considering the chat history. Okay, that is what we are basically going to do. And now if I just go ahead and display this, you'll be able to see this entire thing over here. Right. Now this is perfectly fine till here. Everything looks good. Okay. And we are going on very on point.

Okay, now it's time to go ahead and create the chain. Okay. Now to create the chain, what I will actually do is that again, we will go ahead and write the same question-answer chain. Here we are going to use the "create_stuff_document_chain". Okay. And here we are basically going to use my LM, comma QA, or what is the prompt that I had actually used over here, which is nothing, but it is a contextualized Q-prompt. And instead of this I can go ahead and create the new prompt over here, okay. And I will name it as "QA_prompt". Okay. The same prompt, only nothing different. So I'll use this QA_prompt, okay. It is the same thing, right? So here I'm going to use this particular prompt. And along with this prompt, what I will do in the next step, I will go ahead and create my action. But this time my RAG chain will be having this create retrieval chain. I will go ahead and create this retrieval chain inside this. Instead of just giving the retrieval, I will be giving history-aware retrieval, comma, the question-answer chain. See, sorry, the question-answer chain. If you remember previously when we created this "create_retrieval_chain", what all inputs we specifically gave? See over here we gave the retrieval, we gave question-answer chain. But this time we are giving history-aware retriever and question-answer chain. Now it's time that we can go ahead and use this RAG chain. Okay, now let's go ahead and add it and see what exactly action will basically do and what all things we can specifically do. Okay.

So first of all, what I will do, I will go ahead and import "from langchain.core.messages import AIMessage, HumanMessage". These are the list of messages that we usually append here. I'm just going to go ahead and create my chat_history. And remember the variable name should be same like how we have defined over there, right, in the "MessagesPlaceholder". Then my first question will be something like "question = 'What is self reflection?'". Okay. So from that particular page I'm just going to ask this particular question, "What is self reflection?" Okay. So this basically becomes my question. Okay. Now if I want to execute it, let's say this will be my "response_1", and inside this "response_1" I will go ahead and use the same chain, I will write "response_1 = RAG_chain.invoke(input=question, chat_history=chat_history)". Okay. And let me just go ahead and give my input. The first input will be nothing but "question", and the second input will be nothing but "chat_history". Okay. So here we are going to basically go ahead and use "chat_history" which will be assigned to my chat history. Right. Whatever chat history we are creating over here. And this is the "MessagesPlaceholder" that we are giving. Right. And here only we are going to save all our chat history.

Now along with this, what I will do, since this is a list, right? I'm going to go ahead and append this with every conversation that I have. So here I'm going to write "chat_history.extend([HumanMessage(content=question), AIMessage(content=response_1['answer'])])". Okay. So once it gets appended or extended, expanded basically means it will just get added at the last. Okay. Now similarly, once I execute this, I will go ahead and write my second question. The question two will be that "question_2 = 'Tell me more about it.'" Okay, I'll just go ahead and write it. So this will be my second question that I am going to go ahead and write. Now I will go ahead and write "response_2 = RAG_chain.invoke(input=question_2, chat_history=chat_history)". And here the same thing I'm going to pass right, with respect to this history, I'll get my "response_2". So this will basically be my "response_2". And now I will go ahead and print "response_2['answer']". Okay. Now I think it should be able to understand the context and it should be able to give the answer. Let's see. So self-reflection mechanism: self-reflection is a mechanism that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. He initially asked "What is self reflection?" Then I told "Tell me more about it" and this is what we are basically printing.

Okay, if I go ahead and see with respect to the chat history, you'll also be able to see that first one is this particular message that we got, right? Errors are inevitable. Okay. And second time, when I asked "Tell me more about it", it was able to understand the context and it was able to give us the answer. So, I hope you're able to understand this amazing thing. Okay. Now there are some more things that you can actually do with whenever you are working with LLMs, right? That is adding chat history with respect to session IDs and all that we have already discussed. So let me quickly copy and paste some of the code over here so that you will be able to understand. So if I go ahead and write this, if you remember, this base chat history in our runnable with message history and chat history, right? We used it. So here you can create this particular session. And based on the session ID you will be able to retrieve all the chat history. The output key will be the answer. The input key is nothing but input, right? So here if I go ahead and execute this, here is nothing but it is your entire conversational chain. In short, right. Now I can go ahead and invoke anything based on my session ID. So here I will go ahead and write "conversation_RAG_chain.invoke(input=question, config=config)". This config is going to make sure that it will store all the information with respect to the session ID, and then we can also get the answer, right? So here you can see our task decomposition is the same thing. And here you can probably see the same answer. Okay. Now similarly if I go ahead and use the same session, like I'll go ahead and ask "What are the common ways of doing it?" So I'm actually talking about task decomposition. The reason is why I have actually used the same session ID. You can go ahead and see the answer. Now it will be able to understand that yes, we are fine talking about task decomposition. So here you can say that according to context, most common ways of doing task decomposition includes NN1, something, newline characters, using a large language model. Everything is basically displayed. Let me see one more thing. Did I print? Okay, yes, I printed the right response. So yes, this was all about the conversational Q&A chatbot with memory, right? With chat history. And I hope you are able to understand this particular question. This is how we basically create a history-aware retriever. And you can actually use all these things. Right. So yeah, this was it for my side. I hope you like this particular video and yeah, I will see you all in the next video. Have a great day. Thank you all. Take care. Bye bye.

# **IX)  End To End Q&A Chatbot GEN AI App With**

# **A) Introduction To The Q&A Chatbot**

So finally, I’m excited to implement the first end-to-end Gen AI application, which is nothing but a Q&A chatbot.

I’ll start by giving you a brief architecture of what all things we are specifically going to do in this particular project. Then, from the next video, we’ll begin implementing it step by step.

In this project, some of the tools, APIs, and LLM models that we are going to use include OpenAI’s LLM models. Among them, we will try models like GPT-4, which is multimodal, and GPT-4 Turbo. The entire chatbot will be created as a Streamlit web app, so that you’ll have multiple options to select the model and continue with the interaction.

Apart from OpenAI models, we will also make use of open-source models like LLaMA and Mistral. For example, LLaMA 3 or LLaMA 2 models can be integrated. This way, our chatbot will not just be limited to OpenAI, but will also be able to handle open-source alternatives.

Now, let’s talk about the architecture. First, we are going to create the Streamlit web app. This app will act as the front end for user interaction. Inside the app, we’ll provide options where the user can input queries and select which model to use. The app will then call the OpenAI API (or other model APIs) in the background.

So the flow is simple:

The user enters a query into the Streamlit web app.

The app interacts with the chosen LLM model, such as GPT-4, GPT-4 Turbo, or LLaMA.

The model generates a response.

The response is displayed back to the user in the web app.

But we’re not stopping there. Along with this interaction, we will also log everything into the LangSmith platform. LangSmith will help us with monitoring, debugging, and cost tracking. This means every query, response, and metadata like tokens used will be visible for analysis.

So the flow becomes: user query → Streamlit app → LLM (via OpenAI API or open-source API) → response → LangSmith logs.

Now, let me outline the steps we are going to follow:

Create the project structure and initialize it properly.

Set up environment variables (like API keys). For example, we’ll use a .env file and load it into the app:

"from dotenv import load_dotenv
load_dotenv()"

Add a requirements file (requirements.txt) where we’ll list libraries like streamlit, openai, langchain, and others.

Build the Streamlit web app with an interface that allows selecting models, setting temperature, and adjusting max tokens.

Integrate OpenAI API and LangChain for LLM interaction:

"import openai
response = openai.ChatCompletion.create(
model='gpt-4-turbo',
messages=[{'role': 'user', 'content': 'Hello! Explain Generative AI'}]
)"

Connect LangSmith tracing so that all conversations and costs are logged for monitoring.

Finally, we’ll also focus on deployment of this Streamlit app. That way, anyone can use it directly from a browser without worrying about setup.

To give you a clearer idea, imagine the UI. On the left-hand side, you’ll have an option to input your OpenAI API key. Below that, you’ll have a dropdown to select models like GPT-4 Turbo, GPT-4, or others. You’ll also have sliders to set temperature and maximum tokens.

In the main chat area, you can ask questions like:

"Hi!"
"Please explain what is Generative AI."

And as soon as you press enter, the chatbot will display the model’s response.

At the same time, LangSmith tracing will capture details like latency, cost, and response quality for debugging and monitoring.

So this is the end-to-end project plan: from project setup, environment variables, Streamlit app, OpenAI API integration, LangSmith logging, to deployment. Once this is complete, we’ll gradually move on to more advanced projects like RAG applications and RAG with SQL.

That’s the overview. Let’s go ahead and start implementing this step by step in the next session. Thank you.

# **B) Creating Virtual Environment**

I am excited to implement the first end-to-end Gen AI application, which is a Q&A chatbot. I will give you a brief architecture of what we are going to do in this project, and then from the next video we will start implementing it.

Some of the tools, APIs, and LLM models that we are going to use include OpenAI LLM models. We will use models such as GPT-4, which is a multimodal model, and GPT-4 Turbo. We will create this entirely in a Streamlit web app so that you will have multiple options to select the model and continue.

The second model we will probably use is LLaMA. With LLaMA, we are going to leverage open-source models available such as LLaMA 2 and LLaMA 3. Along with that, we will also use other open-source models like Mistral. All these open-source models will also be integrated to create this Q&A chatbot.

In our project, we will first create the Streamlit web app. This web app will interact with the OpenAI API. With the help of this API, we will interact with various LLM models such as GPT-4, GPT-4 Turbo, and others. Once we interact with these models, we will receive responses for the queries.

We will not stop there. After integrating with the LLM models, we will take one step further and log all interactions in the LangSmith platform. LangSmith will be used for monitoring, debugging, and tracking costs. It will allow us to see details of all the queries, responses, and resource usage.

Here is the flow: the user provides a query, the Streamlit app sends it to the OpenAI API (or open-source models like LLaMA or Mistral), and then we get the response back. All the interactions will also be logged and monitored in LangSmith.

This looks like a simple project, but as we go ahead, we will build more complex projects on top of it, such as RAG applications and RAG with SQL. Many more projects will follow. Right now, we will start with this basic project.

The steps we will cover include:

Creating the project setup.

Setting up environment variables.

Defining requirements in requirements.txt.

Creating the Streamlit web app.

Calling the OpenAI API and integrating LangChain.

Deployment of the project.

These steps reflect how such projects are done in the industry, and I will cover all of them in this project.

To give you an idea of the final outcome: on the left-hand side of the app, you can input your OpenAI API key, select different models such as GPT-4 Turbo or GPT-4, set the temperature value, and define the maximum tokens for the output.

For example, if I ask a question like “What is Generative AI?”, I will be able to get the response displayed in the interface. Alongside, we will also have LangSmith tracing enabled to monitor and debug the process. We will deep dive into these tracings later.

This is the complete overview of what we are going to develop. Let’s go ahead and implement this end-to-end. Thank you.

# **C) Creating Prompt Template And Integrating Open AI API**

So guys the requirement.txt has now been installed.

In this video we are going to develop our Streamlit web app and then we will develop our entire application. Let me quickly close all these things and start.

The first step is to import some of the libraries.
import streamlit as st
Here you can also see my conda environment is there.
import openai
From langchain_openai import ChatOpenAI. Since we are working with OpenAI in the first project, in the upcoming project we will discuss with Ollama.

If you want to work with OpenAI along with LangChain integration, you should start working in LangChain. Do not work independently with different libraries, because LangChain is a framework created in such a way that it can interact with OpenAI APIs, it can interact with open-source LLM models, and it can interact with HuggingFace. All these will be included.

So I am going to quickly import ChatOpenAI. Along with this I will also write:
from langchain_core.output_parsers import StrOutputParser
Then:
from langchain_core.prompts import ChatPromptTemplate

All these libraries will be the basic requirements.

Next, import:
import os
from dotenv import load_dotenv

This is to load all the environment variables. We will initialize this using load_dotenv().

Just to check everything is working fine, inside this project folder I will run:
streamlit run app.py

This works absolutely fine. Nothing is displayed yet, but till here everything looks good. Then press Ctrl + C to stop.

The next step is to implement LangSmith tracking.

Inside this tracking setup, I will write:

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


For LangSmith tracking we need to set this value. Another value we need to set is LANGCHAIN_TRACING = "true". Finally, we set the project name:

os.environ["LANGCHAIN_PROJECT"] = "Q&A chatbot with OpenAI"


Till here, everything looks good.

Now, since we are using Streamlit, we will first define our prompt template. Define it as:

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "{question}")
])


The system prompt means the information given to the LLM about how it should behave. The user prompt is the placeholder for the question. This is a simple Q&A chatbot — the user gives a question, and the system gives the answer.

Next, create a function called generate_response.

Inside this function, whatever user query comes in, it will interact with the OpenAI model and return the response.

The parameters will be:

question: the user’s query.

api_key: the API key passed during runtime for validation (not kept in environment variable).

engine: the LLM model to interact with.

temperature: controls creativity.

max_tokens: maximum number of tokens.

About temperature:

Value between 0 and 1.

0 means less creative, same output for the same question each time.

1 means more creative, varied answers each time.

Inside the function, first set the API key:

openai.api_key = api_key


Then use the LLM model:

llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)


Next, create the chain. The chain combines the prompt template, the LLM, and the output parser. Define:

output_parser = StrOutputParser()
chain = prompt | llm | output_parser


Now invoke the chain:

answer = chain.invoke({"question": question})


Finally, return the answer.

This function is what interacts with the OpenAI LLM models.

In the next video we will create the full Streamlit app and call this function.

That’s it for now. Thank you.

# **D) Creating Streamlit Web App and Integrating Response With OpenAI API**

We are going to continue our discussion with respect to this end-to-end project. In the previous video we created the function called generate_response.

Now we are going to create the entire web app using Streamlit.

First, we will give the app a title. Using Streamlit’s title function:
Q and A Chatbot with OpenAI.

As mentioned earlier, I need to pass my OpenAI API key during runtime. For that, we will use a sidebar in Streamlit. In the sidebar, you can enter your OpenAI API key.

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")


This means no one will see the password directly.

Next, create a dropdown to select various OpenAI models. This will also be inside the sidebar.

llm = st.sidebar.selectbox(
    "Select an OpenAI model",
    ["gpt-4", "gpt-4-turbo", "gpt-3.5"]
)


These are the three models we’ll keep in the dropdown.

There are also two parameters to pass in generate_response: temperature and max tokens. We can set these values with sliders in the sidebar.

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)


Now define the main interface for user input.

st.write("Ask any question:")
user_input = st.text_input("Enter your question")


If the user provides input, we generate a response:

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a query.")


This ensures the response will be displayed on the page when a query is entered. If no input is given, it prompts the user to provide a query.

Let’s quickly revise.

First, we set up LangSmith tracking.

Then we developed the chat prompt template.

We created the generate_response function.

We added user input fields, API key entry, model selection, sliders for temperature and tokens.

Finally, we displayed the response.

Now let’s run the entire application. Clear the terminal and run:

streamlit run main.py


This starts the app. Enter your OpenAI API key, set some values, and ask a question.

For example: What is machine learning?
Press enter, and the response is displayed. It may take some time depending on the API.

You can also see if it is getting tracked in LangSmith. Open LangSmith, sign in, and check the project.

Here you can see:

Chat prompt template

ChatOpenAI

String output parser

The query: “What is machine learning?”

The runnable sequence with the output

The chain shows: first the chat prompt template, then ChatOpenAI interaction, and finally the string output parser with the full response.

Everything looks fine and is working.

In the next video, we will modify this application to work with the Llama models.

Thank you.

# **E) Q&A Chatbot With Ollama And Open Source Models**

We are going to continue the discussion with respect to our LangChain series.

In this video, we are going to create an end-to-end Q&A chatbot using open source models. Specifically, we will be using Llama.

In the previous video, we created this end-to-end chatbot with the help of OpenAI. The best thing was that we made it loosely coupled so we could use any kind of models we want.

Similarly, we are now going to do it with Llama. In the case of Llama, you require a different set of libraries. Instead of using ChatOpenAI, you will use another library called Llama.

First, create app.py. Import from langchain_core.prompts the ChatPromptTemplate. Then from langchain_core.outputs, import StringOutputParser.

Since we want to use Llama, we import from langchain_community.llms the Ollama class. We also import Streamlit as st and import os.

Next, open main.py for the initial setup. Previously, we wrote “with OpenAI,” but now we will write “with Ollama.” These are the initial things required.

Now create the prompt template. Using ChatPromptTemplate.from_messages, define the messages: one system message and one user message. For example, system: “You are a helpful assistant. Please respond to the user queries.” Then the user message is the actual question.

We also define the generate_response function. Unlike OpenAI, here we don’t require any API key because this is open source. Instead of ChatOpenAI, define your Llama model.

To check available models, open the command prompt and run:

ollama run llama2


This installs and runs the model. Similarly, try ollama run gemma:2 or other models. If not available, it downloads them.

On the Ollama website, you can see models like Gemma 2, Llama 3, Phi-3, Mistral, Solar, and others. These are open source models. For example, run:

ollama run phi3


It will download and install Phi-3. Any model you want to use must be installed locally first.

Once installed, you can run:

ollama run mistral


Test it by typing “hi” and see the response.

So we will use one of the models, for example mistral. Note that previous versions may be removed (e.g., Llama 2 replaced by Llama 3).

In the code, set:

model = "mistral"


or whichever model you want, like "llama3".

Now create the chain: prompt → Llama model → output parser. Invoke it with the input text.

Remember, with Llama we are only using open source models. We also don’t need API keys.

Unlike OpenAI, parameters like temperature and max tokens may not always apply. You can keep them in the code, but some open source models may ignore them.

Simplify the app by removing extra conditions. The workflow is the same as with OpenAI:

User input → pass to model → generate response → display in Streamlit.

Now run the app:

streamlit run app.py


Select the model (for example, mistral), enter input, and see the response. For example:

Input: “Hi” → Output: “How can I assist you today?”

Input: “Please talk about generative AI” → Output: full response from the model.

The response is fast.

Check LangSmith to see the tracking. Sign in, and you will see the project logs. The run shows:

Chat prompt template

Llama model call

Output parser

Response with token usage and time taken

Cost is zero since this is local and open source.

So we designed the prompt template, used the open source Llama model, and displayed the response. Everything works as expected.

Later, when more models like Gemma 2 or Llama 3 are downloaded, you can add them as options in the app so the user can select which model to use.

This is how you create an end-to-end Q&A chatbot using Ollama with LangChain and Streamlit.

# **X) RAG Document Q&A With GROQ API And LLama3**

# **A) Introduction To Groq Cloud And LPU Inference Engine**

We are going to continue with a new end-to-end project: a Gen AI project. We will specifically use the Grok AI inferencing engine.

Grok is a platform that provides open source LLM models like Gamma, Llama 3, and Mistral. You can use these models to create an end-to-end generative AI application.

First, understand why Grok. Grok is a fast AI inferencing engine that uses a Language Processing Unit (LPU). In the generative AI field, companies providing amazing inferencing speed in milliseconds will lead. The LPU is a hardware and software platform that delivers exceptional compute speed, quality, and energy efficiency for AI applications like large language models.

LPUs are faster than GPUs because they overcome bottlenecks in compute density and memory bandwidth. You can read more about Grok on their website.

In this video, we will create an API in Grok Cloud and use LM models to start our project. First, go to Grok Cloud and create an API key. There is also a playground where you can test Llama 3 models. For example, typing “hi” will return a response quickly.

To create an API key, click “Create API Key,” give it a project name, and click submit. Copy the API key; it starts with GSK_. Paste this key into your project .env file. Use the environment variable:

GROK_API_KEY


We will use this key to access the LM models available as open source.

Next, we need to install required libraries. In requirements.txt, include LangChain and the Grok library. For example, like we have langchain and huggingface-langchain, add langchain-grok. Then install everything:

pip install -r requirements.txt


Once installed, we are ready to start coding.

Import the necessary libraries. We will use Streamlit for the web app:

import streamlit as st
from langchain_grok import ChatGrok
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import CreateStuffDocumentChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import CreateRetrievalChain
from langchain.vectorstores import VectorStore
from langchain.document_loaders import PyPDFDirectoryLoader


ChatGrok is used to interact with Grok models. For embeddings, you can use Ollama embeddings to avoid paid APIs. RecursiveCharacterTextSplitter is used to split documents for RAG applications. CreateStuffDocumentChain is critical for Q&A applications interacting with external data. ChatPromptTemplate is used to define the prompt structure. CreateRetrievalChain combines the retrieval and response generation.

The vector store and PDF loader allow us to handle external documents. These are the basic imports required for a RAG document Q&A application.

Next, we will load the environment variable for the Grok API key and start building the end-to-end application.

In the next video, we will continue with:

Loading documents

Creating embeddings

Building the RAG chain

Using Grok API to answer queries

This concludes the setup and library imports for now.

Thank you.

# **B) RAG Document Q&A With GROQ API And LLama3**

We continue with the Gen AI project.

We have a folder named research_paper containing two PDFs: Attention is All You Need and Overview of Large Language Models. This folder will serve as an external data source for a document Q&A application. We will ask questions and retrieve answers from these research papers.

We will use ChatGrok for interacting with LM models. Instead of using OpenAI, we will use open-source models, so anyone can execute this project.

First, load environment variables:

from dotenv import load_dotenv
import os

load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")


This loads the Grok API key. Grok provides free API access for a limited number of hits. You can create an account and generate your key.

Next, create the LM model with ChatGrok:

from langchain_grok import ChatGrok

llm = ChatGrok(
    api_key=GROK_API_KEY,
    model_name="Llama-3"  # or Gamma-7B-IT
)


Define the chat prompt template:

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template("""
Hey, answer the questions based on the provided context only.
Please provide the most accurate response.
Context: {context}
Question: {question}
""")


The context placeholder will be filled with document content during retrieval.

Next, create vector embeddings for the research papers. This involves reading PDFs, splitting the text, and storing embeddings in a vector store. We also use Streamlit session state to maintain memory across the app:

from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.document_loaders import PyPDFDirectoryLoader
import streamlit as st

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state["embeddings"] = OllamaEmbeddings()
        st.session_state["loader"] = PyPDFDirectoryLoader("research_paper")
        st.session_state["documents"] = st.session_state["loader"].load()
        st.session_state["text_splitter"] = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state["final_documents"] = st.session_state["text_splitter"].split_documents(
            st.session_state["documents"][:50]  # limit for speed
        )
        st.session_state["vectors"] = VectorStore.from_documents(
            st.session_state["final_documents"],
            st.session_state["embeddings"]
        )

st.button("Document Embedding", on_click=create_vector_embedding)


This reads PDFs, splits text into chunks, converts them into vectors, and stores them in a vector database.

Now, we can use the Grok API to query the vector database:

from langchain.chains.combine_documents import CreateStuffDocumentChain
from langchain.chains import CreateRetrievalChain
import time

user_prompt = st.text_input("Enter your query:")

if user_prompt and "vectors" in st.session_state:
    start_time = time.process_time()
    document_chain = CreateStuffDocumentChain(
        llm=llm,
        prompt=prompt_template
    )
    retriever = st.session_state["vectors"].as_retriever()
    retrieval_chain = CreateRetrievalChain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    answer = retrieval_chain.invoke({"input": user_prompt})
    response_time = time.process_time() - start_time
    st.write(f"Response time: {response_time} seconds")
    st.write("Answer:", answer)
    with st.expander("Document similarity search"):
        for i, doc in enumerate(st.session_state["final_documents"]):
            st.write(doc.page_content)


This implementation:

Loads PDFs and splits them into chunks.

Converts chunks to vector embeddings.

Stores vectors in a vector database.

Uses Grok API with an open-source LM to answer queries.

Provides document similarity context along with answers.

For faster execution, OpenAI embeddings can be used instead of local embeddings, as local processing may take a long time.

Once executed, you can ask questions like:

“What is Transformers?”

“What is a Large Language Model?”

“What is Attention Is All You Need?”

The app retrieves answers from the research papers along with relevant context. You can adjust chunk size and overlap to control the detail level.

This demonstrates a complete RAG-based document Q&A project using Grok AI and open-source LLM models.

# **XI) Conversational Q&A Chatbot- Chat With Pdf Along With Chat History**

# **A) Demo of the Conversational Q&A Chatbot**

In this video and the upcoming series, we will develop an end-to-end Gen AI project: a conversational Q&A chatbot that can chat with a PDF.

We will also maintain conversation history, so the chatbot can refer back to previous interactions. This is a complete end-to-end project.

First, a demo:

After entering the Grok API key, a default session is created. In a real web application, you would track session IDs so each user has a unique session.

Next, drag and drop a PDF file, for example attention.pdf. The text from this PDF will be converted into vectors and stored in a vector database.

Now we can ask questions:

Question: “Tell me about Transformers.”
Response: The transformer generalizes well to English constituency parsing, achieving an F1 score of 91.3, trained on a semi-supervised set.

Question: “Attention is All You Need.”
Response: The attention mechanism is used in transformer architecture to weigh the importance of different input elements when computing the output.

Question: “Which topic were we discussing about?”
Response: Refers to the transformer model and its architecture from previous context.

Question: “Provide a detailed summary of Attention Is All You Need.”
Response: Limited due to the PDF being only two pages.

Question: “Tell me more about transformer and its architecture.”
Response: Detailed description of transformer neural network architecture, including the self-attention mechanism.

Question: “What is self-attention mechanism?”
Response: A self-attention mechanism is a key component of transformer architecture, used to process all information in the input sequence.

All chat history is considered. For example, asking “What is the previous message?” retrieves the previous topic.

The chatbot can summarize the conversation history:

Question: “Provide me a detailed summary of the conversation we had.”

Response: Based on chat history: initially asked about transformer model architecture, self-attention mechanism, and parallelization for processing sequences. It summarizes the conversation accurately, including follow-up questions.

This demonstrates:

Maintaining conversation history using session IDs.

Conversing with PDFs.

Providing detailed answers using context from the vector database.

Generating summaries of the chat history.

In the next video, we will implement the full end-to-end solution. This builds upon the conversational bot concepts discussed in previous modules.

**B) End To End Conversational Q&A Chatbot Implementation**

We are continuing with our end-to-end Gen AI project, which is a conversational Q&A system with PDFs, including chat history. We will implement this step by step. First, we start by importing all necessary libraries. We have already discussed chat history in the previous module, which was about conversational Q&A chatbots.

We begin by importing Streamlit as "import streamlit as st". We will also import some important libraries for creating history-aware retrievers and retriever chains: "from langchain.chains import create_history_aware_retriever, create_stuff_document_chain". The "create_history_aware_retriever" is used to create a retriever with chat history functionalities, and "create_stuff_document_chain" is used to combine documents and send them as context.

We also import our vector store database using Chroma: "from langchain.vectorstores import Chroma". The chat message history module is imported as "from langchain_community.chat_message_history import BaseChatMessageHistory". All these libraries are included in "requirements.txt".

We import the chat prompt template and message placeholder using: "from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder". The message placeholder helps define the session key where all conversation history is stored.

For the language model, we use ChatGrok: "from chatgrok import ChatGrok". We also import HuggingFace embeddings: "from langchain.embeddings import HuggingFaceEmbeddings". We load environment variables with "import os" and "from dotenv import load_dotenv; load_dotenv()". The HuggingFace token is loaded from the environment: "hf_token = os.getenv('HF_TOKEN')".

We initialize our embeddings using HuggingFace: "embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')". To load PDFs, we use the recursive character text splitter and PyPDF loader: "from langchain.text_splitter import RecursiveCharacterTextSplitter; from langchain.document_loaders import PyPDFLoader". For managing chat history in the conversational bot, we import: "from langchain.schema.runnable import RunnableWithChatHistory".

Next, we set up the Streamlit app: "st.title('Conversational RAG with PDF Uploads and Chat History')" and "st.write('Upload PDFs and chat with the content')". We prompt the user for the Grok API key using: "api_key = st.text_input('Enter your Grok API key', type='password')". We check if the API key is provided: "if api_key: chat = ChatGrok(api_key=api_key, model_name='gamma-2')".

We create a session ID input with: "session_id = st.text_input('Session ID', value='default_session')". To manage session state, we initialize a store: "if 'store' not in st.session_state: st.session_state['store'] = {}". This will hold key-value pairs for messages and chat history.

We then handle file uploads with: "uploaded_files = st.file_uploader('Choose a PDF file', type='pdf', accept_multiple_files=False)". For each uploaded file, we save it locally: "temp_pdf = f'temp.pdf'; with open(temp_pdf, 'wb') as f: f.write(uploaded_file.getvalue())". We load documents using PyPDFLoader: "loader = PyPDFLoader(temp_pdf); docs = loader.load(); documents.extend(docs)".

Next, we split and create embeddings for the documents: "text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500); splits = text_splitter.split_documents(documents)". We store these embeddings in Chroma: "vectordb = Chroma.from_documents(splits, embeddings)". The retriever is created: "retriever = vectordb.as_retriever()".

We define a system prompt for contextualizing questions: "contextualize_qa_system_prompt = '''Given a chat history and the latest user question which might reference context in the chat history, reformulate it to a standalone question.'''". We create a ChatPromptTemplate using: "contextualize_qa_prompt = ChatPromptTemplate.from_messages([('system', contextualize_qa_system_prompt), MessagesPlaceholder(variable_name='chat_history'), ('human', '{input}')])".

We then create a history-aware retriever: "history_aware_retriever = create_history_aware_retriever(retriever=retriever, qa_prompt=contextualize_qa_prompt, chat_history=chat_history_placeholder)". We define a QA system prompt: "qa_system_prompt = '''You are an assistant for question answering. Use the retrieved context to answer. Keep it concise, max 3 sentences. If unknown, say you don't know.'''". We create the QA prompt template: "qa_prompt = ChatPromptTemplate.from_messages([('system', qa_system_prompt), MessagesPlaceholder(variable_name='chat_history'), ('human', '{input}')])".

The QA chain is created using: "qa_chain = create_stuff_document_chain(llm=chat, prompt=qa_prompt)". The retrieval chain is wrapped with history-aware retriever: "rag_chain = create_retrieval_chain(retriever=history_aware_retriever, qa_chain=qa_chain)".

We create a session history function: "def get_session_history(session_id): if session_id not in st.session_state['store']: st.session_state['store'][session_id] = BaseChatMessageHistory(); return st.session_state['store'][session_id]".

The conversational RAG chain is initialized: "conversational_rag_chain = RunnableWithChatHistory(rag_chain, get_session_history, input_key='input', history_key='chat_history', output_key='answer')".

Finally, we handle user input: "user_input = st.text_input('Ask your question'); if user_input: session_history = get_session_history(session_id); answer = conversational_rag_chain.invoke({'input': user_input, 'session_id': session_id}); st.write('Answer:', answer)".

If the API key is not provided, we show a warning: "else: st.warning('Please enter the Grok API key')".

To run the app, navigate to the project folder and execute: "streamlit run app.py". The HuggingFace embeddings may take some time to initialize the first time. Once loaded, you can ask questions like "What is Transformers?" or "Provide a detailed summary of Transformers and Attention Is All You Need". The chat history and answers will be displayed along with the session.

This implementation allows uploading any PDF, using chat history, generating concise answers, and summarizing conversations.

# **XII) Search Engine With Langchain Tools And Agents**

# **A) Introduction To Tools And Agents**

So we are going to continue the discussion on creating end-to-end generative AI projects with the help of LangChain. In this video, and in the upcoming series of videos, we are going to develop an end-to-end search engine AI app with the help of tools and agents. Now, these words that you will be seeing — tools and agents — are very important. You may be thinking that this may be a simple search engine project, where you just ask a query to the LLM models and get a response. It is nothing like that. The main intention of doing this specific project is to integrate tools and agents because, with the help of tools and agents, your generative AI application becomes more powerful in solving complex problems.

Now, what exactly are we going to do in this project? Let’s say a user asks a query. First, we need to understand what exactly tools are and what exactly agents are. Tools are interfaces that an agent chain or LM can use to interact with the world. They combine a few things like the name of the tool, description of what the tool is, and how to use it. For example, one tool could be an OpenAI LLM model, like GPT-4. This model is trained up to December 2023. So, whatever question you ask with respect to the data available until December 2023, you can get a response easily.

But now, in this world, we also require current information. There are a lot of changes happening right now — for example, current news, current weather, updated documentation, or recently published research papers. If the generative AI app only uses the LM model, it cannot fetch this latest information. That’s where tools come in. These tools allow the LM to interact with external sources and retrieve current content. For example, tools like RCF (for research papers) or Wikipedia allow you to query the latest content directly. You can also create your own custom tools, like a document Q&A tool that allows you to query your uploaded documents.

So, whenever we create this search engine, it will not be dependent only on the LM model. Instead, if the user asks for information not available in the LM model, the LM can interact with these tools and fetch the answer. This allows your search engine AI app to provide up-to-date and accurate responses.

Now, to make all this work, we also need another component called agents. Agents are responsible for orchestrating the tools. The core idea of an agent is to use a language model to choose a sequence of actions to take. For example, an agent can decide: first, check the LM model, then query RCF for research papers, and finally check Wikipedia for additional information. Agents manage the workflow and ensure the LM interacts with the tools efficiently.

LangChain provides multiple built-in tools like Wikipedia, Yahoo Finance, YouTube, Weather API, and more. Some of these are free, while others require an API key. Similarly, LangChain allows you to create different types of agents that can orchestrate these tools in a sequence of actions to solve complex queries.

In this project series, we will first explore tools, how to create and use custom tools with LangChain, and then discuss agents. Finally, we will combine all of these to build an end-to-end search engine AI app that can answer complex questions using both LLM and external tools in a seamless workflow.

So, this was it for this video. I hope you understood the concepts of tools and agents, how they work, and why they are essential for building advanced generative AI applications. In the next video, we will dive into the practical implementation of tools, agents, and how to integrate them into a working search engine app.

**B) Creating Tools Using Langchain**

So we are going to continue the discussion with respect to tools and agents in the search engine. First of all, I will show you some important functionalities where I will create my own custom tools, and then I will also use the inbuilt tools that are available in the engine.

Some of the inbuilt tools available in the engine are Wikipedia and RCF, which we have already discussed. You can also see other tools like Google Finance, Google Places, Google Search, and more. For example, if we want to use Wikipedia, we need to first install the Wikipedia library and then implement it.

Let me show you with code. First, I will select the kernel and set it up for research purposes. This tool will interact with the RCF website to fetch information. We need to ensure that the libraries are installed, so I will install both RCF and Wikipedia. In the terminal, I run pip install -r requirements.txt. After installation, we can start creating tools.

First, I will create a Wikipedia tool and an RCF tool. We import the required modules:

from langchain_community.tools import RCFQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


We create the API wrapper for Wikipedia:

wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
WikipediaQueryRun = wiki
print(wiki.name)


This gives us the Wikipedia tool. Similarly, for RCF:

rcf_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
RCF = RCFQueryRun(api_wrapper=rcf_wrapper)
print(RCF.name)


To combine both tools, we create a list:

tools = [wiki, RCF]


Now, to run these tools, we need an LM model along with a chain or agent to orchestrate them.

Next, we can create our own custom tools. For example, we can create a RAG (retrieval-augmented generation) tool using a web-based loader, text splitter, and embeddings. We import the necessary modules:

from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


We load a specific webpage, split the document, create a vector store, and a retriever:

loader = WebBaseLoader(url="YOUR_PAGE_URL")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

vector_db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
retriever = vector_db.as_retriever()


We can convert this retriever into a tool:

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="LangSmithSearch",
    description="Search any information about LangSmith"
)
print(retriever_tool.name)


This becomes our custom retrieval tool. Now we have both inbuilt and custom tools ready. We can combine them in a list and pass them to agents or LM models to run queries.

In the next video, we will see how to execute all these tools together with agents and LM models to build a fully functional search engine.

**C) Executing Tools And LLM with Agent Executors**

So we are going to continue the discussion with respect to tools and agents. In the previous video, we have seen how to create tools and even custom tools. Now it’s time to combine these tools with our LM model and execute them using something called an agent executor. This is where we will learn about agents.

First, let’s set up an LM model. I will be using the Grok API for this. Make sure your .env file has your OpenAI API key. We can load it in Python like this:

import os
openai_api_key = os.getenv("OPENAI_API_KEY")


Since we are using Chat Grok API, we will import it:

from langchain_grok import ChatGrok


I’ll be using Llama 3 as the LM model. The OpenAI API key is required here because we are using OpenAI embeddings for the documents. Once the environment variable is set, we can initialize the LM model.

Next, we create a prompt template. LangChain provides a hub where prebuilt prompts are available. For example:

from langchain import hub
prompt = hub.pull("OpenAI_function_agent")
print(prompt.messages)


This prompt contains system message templates, chat history placeholders, and human message templates. Although there are multiple ways to create prompts, using ChatPromptTemplate is generally the best practice.

Now, we combine the prompt with the LM and the tools we have created using an agent. We import and create the agent like this:

from langchain.agents import create_openai_tools_agent

agent = create_openai_tools_agent(
    llm=llm_model,
    tools=tools,
    prompt=prompt
)


This creates the agent chain combining the LM, tools, and prompt.

To execute the agent, we use an AgentExecutor:

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


Setting verbose=True lets us see the interaction details. Now we can invoke the agent executor with an input query:

agent_executor.invoke("Tell me about LangSmith")


You will see that the executor automatically interacts with the correct tool to fetch information. For example, asking about LangSmith opens the LangSmith search tool, while asking about “machine learning” pulls from Wikipedia. The executor also maintains chat history so that subsequent queries are aware of previous context.

We can try another example, such as asking about a research paper in RCF:

agent_executor.invoke("What is the research paper 'Attention is All You Need' about?")


The agent will route the query to the correct tool and return the results.

This demonstrates how agents intelligently combine LM models with multiple tools to create a functional search engine. In the next video, we will combine everything to implement a full end-to-end search engine project.

Make sure to review the documentation to understand all options for AgentExecutor, how to create and combine tools, and how to handle multiple tools efficiently.

**D) End To End Search Engine GEN AI App using Tools And Agent With Open Source LLM**

We are now going to continue the discussion on building an end-to-end AI search engine application with the help of LangChain tools and agents. In the previous session, we learned how to create custom tools, execute them using agents, and even explored some inbuilt tools supported by LangChain. Now, let’s move ahead and start working on the actual project setup.

Inside my project folder, I’ve created a file named "app.py", which will serve as the main driver of this search engine. To begin with, the very first step is to import the required libraries. We start by importing Streamlit using "import streamlit as st". Alongside this, we also need several other important imports. For instance, we will use "from langchain_grok import ChatGrok" to access Grok models. Additionally, we will rely on "from langchain_community.utils import RCFWrapper, WikipediaWrapper" to make use of ready-made wrappers. These wrappers allow us to interact with APIs like Arxiv and Wikipedia with minimal effort.

Apart from this, we will also add some tools from "langchain_community.tools". Specifically, we will import "ArxivQueryRun" and "WikipediaQueryRun" for fetching results from Arxiv papers and Wikipedia articles. To extend our application with web search capabilities, we will also use "DuckDuckGoSearchRun" which will enable internet-wide searches directly from our app. With these imports, our search engine is now capable of handling multiple types of queries.

Once these imports are done, we need to bring in the agents framework by using "from langchain import agents". Agents play a critical role here as they act as the reasoning layer that decides which tool to call depending on the user’s query. Along with agents, we also bring in callbacks to make things interactive. For this, we import "from langchain.callbacks import StreamlitCallbackHandler". This callback handler ensures that the intermediate thoughts and actions of the agent are displayed live within the Streamlit app, giving a transparent view of how the model reasons through a query.

Next, we also import "import os" and "from dotenv import load_dotenv". These help us manage environment variables, especially if we need API keys. However, in our design, we will also provide an option for the user to directly enter their API key in the sidebar of the Streamlit app. For that, we can add "st.sidebar.text_input('Enter your Grok API key')" which allows the user to provide credentials without loading them from .env.

Now that the setup is ready, let’s initialize the tools. For example, we can create an Arxiv tool by calling "arxiv = ArxivQueryRun()", a Wikipedia tool using "wikipedia = WikipediaQueryRun()", and a search tool with "search = DuckDuckGoSearchRun()". These are all wrapped under meaningful variable names to ensure that our agent understands which tool is which.

With the tools in place, we move to the UI part of the application. Using Streamlit, we first define a title with "st.title('Langchain Chat with Search')". This sets the interface heading. As a reminder, in this example, we are using "StreamlitCallbackHandler" to display the thoughts and actions of the agent within the app. This helps us visualize how the agent decides when to call Arxiv, Wikipedia, or DuckDuckGo based on the query provided by the user.

Finally, once the imports, wrappers, and tools are ready, we will proceed to initialize the agent by combining the tools with the LLM and then wrapping them with "agents.initialize_agent()". The agent will then be capable of receiving a user query, reasoning which tool to use, executing the tool, and returning a final answer—all while showing the intermediate reasoning steps inside Streamlit.

# **XIII) Gen AI Project-Chat With SQL DB With Langchain SQL Toolkit and Agentype**

**A) Demo of the Project**

Hello guys! In this series, we are going to continue our LangChain journey, and in the upcoming videos, we will develop an amazing end-to-end project: Chat with SQL DB. The idea is to build an application that can directly interact with a SQL database, generate queries automatically, and return results conversationally.

First, let me give you a quick demo of what we will achieve. We will start by connecting to a simple SQLite3 database, say "student.db", which I’ve already created locally. Using LangChain and agents, we’ll allow natural language questions to be converted into SQL queries and executed on this database. For authentication with open-source models such as Llama-3, Gemma-2, or others supported by Grok, we will use the "grok API key".

When you run the application, you will first paste your Grok API key. For example, using "st.sidebar.text_input('Enter your Grok API key')" in Streamlit will let you provide it directly. Once the app starts, you’ll see a prompt like “How can I help you?”. At this point, you can ask something like "display all the records in the student table". As soon as you press Enter, the agent automatically analyzes the available tables, constructs the proper query, and retrieves the data. Behind the scenes, the agent will figure out that the right query is "SELECT * FROM student;" and return the results, such as Krish, John, Mukesh, Jacobs, and others.

This happens because LangChain agents inspect the database metadata first. For example, they call functions like "SQLDatabaseToolkit.get_table_names()" internally, identify available tables, then proceed to generate the correct query. All of this reasoning and query creation is performed automatically by the LangChain agent.

Now, the best part is that this doesn’t stop at SQLite. You can also connect to your own SQL database like MySQL. In that case, you need to provide the connection details. For example: "mysql_host = 'localhost'", "mysql_user = 'root'", "mysql_password = '12345'", and "mysql_database = 'student'". With these, you can set up a connection string like "mysql+mysqlconnector://root:12345@localhost/student". Once you press Enter, the app will be connected to your MySQL database instead of SQLite.

From here, you can ask the same kinds of questions. For example, "display all the records in the student table" or "display the record where the student name is Krish". The agent will generate queries like "SELECT * FROM student WHERE name = 'Krish';" and return the results directly. What’s happening internally is that the agent uses the SQL DB list tables tool to get the schema (e.g., ['student', 'student_info', 'search_info']), checks which table contains the relevant column, and then employs the SQL DB query checker tool to validate the generated query before executing it.

This ensures that the final query is syntactically correct and runs without issues. Once validated, the query is executed, and the output is shown in the Streamlit interface. You will see the action, the action input, and finally the output—all in a clean and interactive way.

The most powerful aspect of this project is that we will be using the "SQLDatabaseToolkit" provided by LangChain. This toolkit comes with predefined tools and functionalities to handle SQL database queries. When combined with agents, we essentially get an SQL Agent Toolkit, which enables natural language to SQL translation seamlessly. Setting this up involves importing the toolkit with "from langchain.agents.agent_toolkits import SQLDatabaseToolkit", passing your database connection, and then initializing your agent with "initialize_agent()".

So to summarize: in this project, the agent will first list tables ("SQL DB list tables"), inspect schemas, validate queries with "SQL DB query checker", generate the final SQL command (like "SELECT * FROM student;"), and then return the results. All of this happens automatically when you type a natural language question.

This makes for a fantastic hands-on project: a conversational interface to any SQL database, powered by LangChain agents and toolkits. In the upcoming videos, we will go step by step—setting up the connection, configuring the toolkit, and wiring everything with Streamlit to make it interactive. I hope you’re excited, and I’ll see you in the next part where we actually start coding this project.

**B) Preparing the Data For SQlite3 Database**

Hello guys! Let’s continue with this project and implement it step by step. In this session, we will start by creating our data in an SQLite3 database and also learn how to run queries on it. The prerequisite for this is having basic SQL knowledge and being able to set up SQL tools like MySQL Workbench on your system. Even if you don’t have MySQL installed, you can follow along with SQLite3, which comes built-in with Python.

First, let’s create a file named "sqlite.py". Inside this file, we will write all the code required to create a database and a table in SQLite3. To begin, we need to import the SQLite3 module using "import sqlite3". Since SQLite3 is built into Python 3.x, you don’t need any additional installations. Next, we create a connection to the database by writing "connection = sqlite3.connect('student.db')" and then create a cursor object with "cursor = connection.cursor()". This cursor will be used to execute SQL commands like creating tables and inserting data.

Now, we will define the table structure. We create a student table with columns for name, class, section, and marks. The SQL command for this is: "table_info = 'CREATE TABLE student (name VARCHAR(25), class VARCHAR(25), section VARCHAR(25), marks INTEGER)'". We then execute it using "cursor.execute(table_info)". This will create the table inside the database.

After creating the table, we insert some initial records. For example, we can use commands like "cursor.execute('INSERT INTO student VALUES (\"Krush\", \"Data Science\", \"A\", 90)')" and similarly add records for John, Mukesh, and Jacobs. Each cursor.execute() statement inserts a row into the table. Once all rows are added, we can retrieve and display them using "cursor.execute('SELECT * FROM student')" and iterating through the results: "for row in cursor.fetchall(): print(row)".

To make sure all changes are saved, we commit them with "connection.commit()" and always close the connection using "connection.close()". This ensures the database is properly updated and prevents any locks or corruption.

Once the code is ready, we can run it from the terminal. Navigate to the folder containing "sqlite.py" and run "python sqlite.py". If everything is correct, you will see all the inserted records printed, confirming that the student.db database has been created successfully with the proper table and data. Common errors like "cursor has no attribute" can occur if the cursor is not properly initialized. Always make sure you use "cursor = connection.cursor()".

At this point, we now have a fully prepared SQLite database with a table called "student" and multiple records. In the next step, we can connect this database to our LangChain agent for querying and performing interactive operations. Similarly, you can also replicate this process for a standalone SQL server installed locally, such as MySQL or SQL Server, by adjusting the connection parameters.

This approach ensures that you have a structured dataset ready for your AI-powered SQL chat application. Remember, having some familiarity with SQL databases is essential, as it helps you understand table creation, query execution, and handling results efficiently.

**C) Preparing The Data For My SQL Database**

Hello guys! Let’s continue our discussion on this project. In this video, I’ll give you an idea of how to install MySQL Workbench so that you can work with MySQL databases for our AI-powered SQL chat project.

First, go to Google and search for "MySQL Database Workbench". You should see a button called "Download MySQL Workbench". Once you click on it, select your operating system—Windows, Linux, Fedora, Mac OS, etc. For example, if you are using Windows, just click on the Windows option. After that, you might see a login prompt, but you can skip it by selecting "No thanks, just start my download". The download will start automatically.

Once the installer is downloaded, double-click it and follow the installation steps. Keep clicking "Next" and make sure to check all the available checkboxes during installation. After completing the installation, MySQL Workbench will be ready to use. For additional help, you can also check out the YouTube playlist I have created on complete MySQL tutorials. The link will be shared in the video description so you can follow along.

After the installation, open MySQL Workbench and establish a SQL connection to your database. Once connected, you can start executing queries in your database. For example, to prepare your student table, first you might want to drop it if it already exists using "DROP TABLE student;". Then, you can create the table with a command like: "CREATE TABLE student (name VARCHAR(25), class VARCHAR(25), section VARCHAR(25), marks INTEGER);" and execute it. This ensures your table is ready to store data.

To view the records in your table, simply run "SELECT * FROM student;". This will display all existing records in the table. The SQL commands in MySQL Workbench are similar to what we used in SQLite earlier, so the workflow is consistent. Once the table is created and the data is verified, we are ready to move on to the next step of our project, which is building the AI SQL chat interface using LangChain agents.

Remember, working with MySQL requires some knowledge of SQL, so it’s helpful to go through tutorials or my YouTube playlist to understand how to write and execute queries. Once your student table is set up and verified in MySQL, we can proceed to integrate it with our project in the next video.

That’s it for this setup session—once your database and table are ready, we are all set to start building the project. See you in the next video!

**D) Creating the Streamlit Web app and Configuring the Databases**

Hello guys! Now finally we will start the development of our application. First, we will create our Streamlit application file, "app.py". In this file, we will begin by importing some basic libraries required for the project. First, import Streamlit using "import streamlit as st". Streamlit will be used for the entire front-end of our application.

Next, we will import "Pathlib" using "from pathlib import Path" to help us manage absolute file paths in our project. In LangChain, we have an agent system that will allow us to create a SQL agent, which will run queries on our database. The SQL agent is created using "from langchain.agents import create_sql_agent". This function allows us to create a SQL agent from an LM and toolkit or a database, and it will be central to how our application interacts with SQL.

We also need the SQL Database Toolkit from LangChain, which simplifies interacting with SQL databases. For connecting to SQL databases like SQLite or MySQL, we will use "from sqlalchemy import create_engine" from SQLAlchemy. SQLite itself will be imported using "import sqlite3". Since we are using Grok for our open-source language model, we will also import "from langchain_grok import ChatGrok" to interact with the LM.

Next, we set up the Streamlit page configuration using "st.set_page_config(page_title='LangChain Chat with SQL DB', page_icon='🪶')" and add some warning messages about potential prompt injection vulnerabilities. For safety, it is recommended to use a database role with limited permissions.

We then define two global variables for our database options: "local_db" for the SQLite database and "use_mysql" for MySQL. On the sidebar, we create a radio button using "st.sidebar.radio('Choose the DB you want to chat', options=radio_opt)" to let the user select between "Interact with student.db" and "Connect to your MySQL database".

If the user selects MySQL, we require additional inputs for "MySQL host", "MySQL user", "MySQL password", and "MySQL database". This can be implemented using:

mysql_host = st.sidebar.text_input("Provide MySQL hostname")
mysql_user = st.sidebar.text_input("Provide MySQL user")
mysql_password = st.sidebar.text_input("Provide MySQL password", type="password")
mysql_db = st.sidebar.text_input("Provide MySQL database")


For Grok API access, we ask the user for their key using:

api_key = st.sidebar.text_input("Grok API Key", type="password")


We then check that both the DB URI and API key are provided. If not, we display messages using:

if not db_uri:
    st.info("Please enter the database information and URI")
if not api_key:
    st.info("Please add the Grok API key")


Next, we initialize the LM model with Grok using:

llm = ChatGrok(api_key=api_key, model_name="llama-3-8b-192", streaming=True)


Now we define a function "configure_db" to set up the database connection. This function is decorated with "@st.cache_resource(ttl=7200)" to cache the connection for two hours. It accepts parameters like db_uri, mysql_host, mysql_user, mysql_password, and mysql_db.

If the SQLite option is selected, we determine the absolute path using:

db_file_path = Path(__file__).parent / "student.db"
db_file_path = db_file_path.resolve()


We then create a connection using:

creator = lambda: sqlite3.connect(db_file_path, uri=True)
db = SQLDatabase(engine=create_engine("sqlite://", creator=creator))


If the MySQL option is selected, we first check that all necessary parameters are provided, then create the connection using SQLAlchemy and MySQL connector:

if not mysql_host or not mysql_user or not mysql_password or not mysql_db:
    st.error("Please provide all MySQL connection parameters")
else:
    db_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
    db = SQLDatabase(create_engine(db_uri))


Finally, we call this "configure_db" function based on the user's selection. For MySQL:

db = configure_db(db_uri, mysql_host=mysql_host, mysql_user=mysql_user, mysql_password=mysql_password, mysql_db=mysql_db)


And for SQLite:

db = configure_db(db_uri)


At this point, the database connection is fully configured. In the next video, we will create the SQL toolkit that allows the LM model to generate queries and interact with the database automatically. This toolkit will convert natural language prompts like "Display all records in the student table" into SQL queries, fetch results, and return them via the Streamlit interface.

So far, we have completed the database connection setup for both SQLite and MySQL, along with integrating the LM model. The next step is to create the toolkit and agents that will handle queries.

**E) Integrating Web App With Langchain SQL Toolkit And Agenttype**

Hello guys! We will continue with our end-to-end project. So far, we have configured the LLM and the database connection, whether it is SQLite or MySQL. Now it’s time to run the application and test if everything is working.

To run the Streamlit app, use the command:

"streamlit run app.py"


When the page loads, you will see both options for database selection. Selecting either option will display the required input fields, including the Grok API key. Once you provide the API key, the configuration is complete.

The next step is to create a text input box for user queries so that we can interact with the database. For this, we need to work with a SQL Toolkit provided by LangChain.

First, we create the SQL database toolkit using our configured database object "db":

"toolkit = SQLDatabaseToolkit(db=db)"


Next, we create a SQL agent using "create_sql_agent" and pass in the LLM model, the toolkit, verbosity settings, and agent type. We use "zero-shot-react-description" as the agent type:

"agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type='zero-shot-react-description')"


We then set up the Streamlit session state to maintain chat history. If the messages key does not exist, we initialize it with a default assistant message:

"if 'messages' not in st.session_state:"
"    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! How can I help you?'}]"


A clear history button is also provided:

"if st.sidebar.button('Clear message history'):"
"    st.session_state['messages'] = []"


Next, we display all previous messages in the chat interface:

"for message in st.session_state['messages']:"
"    st.chat_message(message['role']).write(message['content'])"


To take user input, we use a chat input box:

"user_query = st.chat_input('Ask anything from the database')"


Once the user submits a query, we append it to the session state and display it on the interface:

"st.session_state['messages'].append({'role': 'user', 'content': user_query})"
"st.chat_message('user').write(user_query)"


We then create a Streamlit callback handler to show the chain of thought while processing queries:

"streamlit_callback = StreamlitCallbackHandler(container=st.container())"


Finally, we run the agent to get the response, and append it to the session state as an assistant message:

"response = agent.run(user_query, callbacks=[streamlit_callback])"
"st.session_state['messages'].append({'role': 'assistant', 'content': response})"
"st.chat_message('assistant').write(response)"


Now we can run the app again using:

"streamlit run app.py"


We first test with the SQLite database (student.db) by providing the Grok API key. Messages can be cleared using the Clear message history button. For example, queries like:

"Show me all records from the student table"


or

"Display all records from the student table whose marks are greater than 45"


will automatically generate SQL queries and fetch the results. The chain-of-thought reasoning of the model is visible thanks to the Streamlit callback handler.

Next, we can connect to the MySQL database. The required inputs are:

"MySQL host: localhost"
"MySQL user: root"
"MySQL password: 12345"
"MySQL database: student"
"Grok API key: <your API key>"


If you see an error like "No module named mysql", install the required libraries by updating "requirements.txt" and running:

"pip install -r requirements.txt"


This should include "SQLAlchemy" and "mysql-connector-python" for MySQL support.

Once connected, queries like:

"Display all records from student table"


or more complex queries like:

"Display all records whose marks are greater than 35 and less than 60"


will work automatically. The SQL Toolkit from LangChain converts these natural language queries into SQL commands and returns the results.

This setup allows you to run both simple and complex queries, including nested queries, directly from a Streamlit interface with real-time LLM support.

# **XIV) Text Summarization With Langchain**

**A) Introduction To text summarization With Langchain**

We are continuing our discussion on LangChain, and in this video (and the upcoming series), we will focus on text summarization.

Text summarization in LangChain allows you to summarize content from any data source, including structured files (like CSVs, Excel) and unstructured content (like PDFs, website text, YouTube transcripts, tweets, or Wikipedia pages).

There are three main summarization techniques in LangChain:

Stuff – Concatenates all documents into a single prompt for summarization.

MapReduce – Splits the document into batches, summarizes each batch, and then combines the summaries.

Refine – Iteratively updates a rolling summary by processing documents in sequence.

In this video, we focus on basic implementation using Stuff, MapReduce, and Refine. Later, we’ll build an end-to-end project to summarize both structured and unstructured content.

Step 1: Setting Up Environment

Import necessary libraries and load environment variables:

"import os"
"from dotenv import load_dotenv"
"load_dotenv()"


We will use Grok API to connect to open-source LLM models, eliminating the need to use OpenAI.

Step 2: Understanding LangChain Schema

LangChain provides a schema to structure messages:

"from langchain.schema import AIMessage, HumanMessage, SystemMessage"


"AIMessage" – Response from the LLM.

"HumanMessage" – User queries.

"SystemMessage" – Instructions for the model on how to behave.

For summarization, we provide a system message to instruct the model and a human message for the specific content to summarize.

Step 3: Summarizing a Speech

We take a long speech (for example, by Prime Minister Narendra Modi) and summarize it:

"chat_messages = ["
"    SystemMessage(content='You are an expert in summarizing speeches'),"
"    HumanMessage(content=f'Please provide a short and concise summary of the following speech: {speech}')"
"]"


"speech" is the variable containing the original text.

The LLM will process this chat message list and provide a summary:

"summary = llm(chat_messages).content"


"AIMessage" contains the full response.

You can check token usage:

"num_tokens = llm.get_num_tokens(chat_messages)"


For example, 895 input tokens were summarized to 108 output tokens.

Step 4: Using LM Chain with Prompt Templates

When text is large, or you want custom formatting, use LM Chain with Prompt Templates:

"from langchain.chains import LLMChain"
"from langchain.prompts import PromptTemplate"


Create a generic prompt template:

"generic_template = '''"
"Write a summary of the following speech: {speech}"
"Translate the precise summary to {language}"
"'''"


Define input variables:

"prompt = PromptTemplate(input_variables=['speech', 'language'], template=generic_template)"


Format the prompt with values:

"formatted_prompt = prompt.format(speech=speech, language='French')"


This generates a fully formatted prompt including translation instructions.

Token count increases slightly because of extra instructions.

Step 5: Running LM Chain

Combine the LLM and the prompt template into an LM Chain:

"lm_chain = LLMChain(llm=llm, prompt=prompt)"


Run the chain to get the summary:

"summary = lm_chain.run(speech=speech, language='French')"


You can change language='Hindi' or any other language.

This method is effective for moderately sized documents.

Step 6: Handling Large Documents

If the document is huge (like PDFs with 25+ pages), token limits may exceed LLM capacities.

In such cases, use Stuff, MapReduce, or Refine document chains:

Stuff Document Chain – Concatenate all documents and summarize in one prompt.

MapReduce Document Chain – Split document, summarize each part, then summarize the combined summaries.

Refine Document Chain – Update a rolling summary iteratively for each document.

Each technique is useful for different document sizes and summarization needs.

We will demonstrate these techniques with 5–6 page PDFs in the next video.

Summary

Text summarization in LangChain can handle structured and unstructured content.

You can use chat message lists or LM Chain with prompt templates.

Token limits matter: for large documents, switch to Stuff, MapReduce, or Refine chains.

System messages guide the LLM, human messages provide content, and AI messages hold responses.

This sets up a foundation for building an end-to-end summarization project with LangChain.

**B) Stuff Chain And Map Reduce Text Summarization Indepth Intuition**

We are going to continue the discussion with respect to text summarization. In this video, we are going to see the most important text summarization techniques that we can use with LangChain. The first one is called the stuff document chain text summarization, and we will discuss one by one what exactly it is. First of all, we will go ahead with text summarization using the stuff document chain. The second technique is called MapReduce summarization. The most important thing about MapReduce is that it is specifically useful for larger files. For example, if I have a PDF file with 100 pages, I cannot directly use the stuff document chain.

In the MapReduce summarization technique, there are two important types: one using a single prompt template and the second using multiple prompt templates. We will discuss this with examples. The third technique you will see is called refine chain summarization. We will first understand what the stuff document chain is, what its limitations are, why we use MapReduce, and when we should use refine chain summarization.

Stuff document chain summarization is the most basic type of summarization technique. Let's say we have a PDF as our external data source; it could be a PDF, a text file, a website, or a web URL. First, we read the entire content, then use a prompt template, for example, "prompt_template = 'Please summarize the following document: {text}'". We pass this along with the prompt template to the LLM model, which generates the output summary.

In the stuff document chain, if the PDF contains multiple documents—for instance, ten documents—these documents are combined into a single text string and sent to the prompt template. The placeholder in the prompt template, for example "{text}", is replaced by this combined text. This works fine for small documents, but for very large documents or websites with thousands of documents, the combined text may exceed the context size of the LLM. For instance, GPT-3.5 has a context limit of 4096 tokens. If the document is too large, sending all content at once will not work properly.

To address this limitation, we use the MapReduce summarization technique. In MapReduce, the document is first divided into smaller chunks, for example, "chunks = [doc1_chunk, doc2_chunk, doc3_chunk, ...]". Each chunk is passed to a prompt template along with the LLM model to get individual summaries, for example, "summaries = [llm(prompt_template.format(text=chunk)) for chunk in chunks]". After getting all individual summaries, we combine them and pass them to another prompt template to generate the final summary: "final_summary = llm(final_prompt_template.format(text=' '.join(summaries)))".

MapReduce can be implemented using a single prompt template, which is applied to all chunks, or multiple prompt templates, where different prompts are used for intermediate and final summaries, such as extracting titles or motivational quotes from the combined summaries. The term MapReduce comes from its workflow: we first map by summarizing each chunk individually, and then reduce by combining all intermediate summaries into a final summary.

The process can be visualized as follows: if the document fits in the LLM context window, we use the stuff document chain—combine all documents, pass them to the prompt, and get the final summary. If the document is too large, we split it into chunks, summarize each chunk, combine the summaries, pass them to a final prompt, and get the final summary.

For implementation, if the document fits in the LLM context, we can use: "combined_text = ' '.join(all_documents)" and "summary = llm(prompt_template.format(text=combined_text))". For large documents using MapReduce: "chunks = [doc1_chunk, doc2_chunk, doc3_chunk, ...]", "summaries = [llm(prompt_template.format(text=chunk)) for chunk in chunks]", and "final_summary = llm(final_prompt_template.format(text=' '.join(summaries)))".

In summary, if the document fits in the context window, we use the stuff document chain. If it is too large, we divide it into chunks, summarize each chunk, and combine the results using MapReduce. In the next video, we will see the practical implementation of these techniques.

**C) Stuff And Map Reduce Summarization Impelmentation**

So we are going to continue the discussion with respect to text summarization. Already in our previous video, we have seen the theoretical intuition behind stuff document chain and map reduce text summarization techniques. In this video, I will show you the practical implementation and how you can specifically implement it.

First, we will go with the stuff document chain text summarization technique. Here I have written a code using "from langchain.document_loaders import PyPDFLoader" where we will be loading a specific PDF. This PDF that I’ve taken is called "FPGA_speech.pdf". Inside this particular PDF, there is an amazing speech given by our former president, Dr. APJ Abdul Kalam, a famous personality. We will read this PDF and summarize the entire text inside it.

After executing the PDF loader, you will see many documents are created because each page or section is considered a separate document. We can inspect them using "documents". Once the documents are ready, we move on to summarization using the stuff document chain. First, we need to create a prompt template. For that, we can write something like "prompt_template = '''Write a concise and short summary of the following speech: {text}'''" and define our input variable as "text". Then we can create the prompt object using "from langchain.prompts import PromptTemplate" with "prompt = PromptTemplate(template=prompt_template, input_variables=['text'])".

Next, we implement the stuff document chain using LangChain. We import "from langchain.chains import load_summarize_chain" and create a chain like "chain = load_summarize_chain(llm=llm, chain_type='stuff', prompt=prompt, verbose=True)". The key idea with the stuff chain is that all documents are combined, passed to the prompt template, then forwarded to the LLM model, and finally we get the summarized output. To get the summary, we execute "output_summary = chain.run(documents)". The result is a concise summary of Dr. Kalam’s speech.

Now, for larger documents, we use the MapReduce summarization technique. First, we split the document into chunks using "from langchain.text_splitter import RecursiveCharacterTextSplitter" and create a text splitter like "text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)". Then we split the documents with "final_documents = text_splitter.split_documents(documents)". These chunks are individually summarized using a prompt template for each chunk, for example "chunk_prompt = '''Please summarize the following speech: {text}'''" and "chunk_template = PromptTemplate(template=chunk_prompt, input_variables=['text'])".

Once we have all the chunk summaries, we create a final prompt template for combining them, like "final_prompt = '''Provide the final summary of the entire speech with important points. Add a motivational title. Start with an introduction and present summary in numbered points: {text}'''" and "final_template = PromptTemplate(template=final_prompt, input_variables=['text'])". Then we create the MapReduce summarize chain using "map_reduce_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=chunk_template, combine_prompt=final_template, verbose=True)". Finally, we execute "output_summary = map_reduce_chain.run(final_documents)" to get the final summarized output of the entire speech, including the motivational title and numbered points.

This way, the stuff document chain works best for small documents that fit in the LLM context window, while MapReduce is ideal for larger documents. In MapReduce, each chunk is summarized individually and then combined with another prompt template to get the final output.

In our next video, we will discuss the refine chain summarization technique and highlight the differences between stuff, MapReduce, and refine summarization. This concludes the practical implementation of stuff and MapReduce summarization techniques using LangChain.

**D) Refine Chain Summarization Intuition And Implementation**

So we are going to continue our discussion with respect to text summarization. We have already seen stuff document chain summarization and MapReduce chain summarization. In this video, we are going to explore refine chain summarization. Refine chain is a slight modification compared to MapReduce. Let’s say we have an entire document that we have divided into smaller chunks: chunk one, chunk two, and so on.

Whenever we use refine chain summarization, we take the first chunk, pass it to a prompt template along with the LLM model, and get the summarization output for that chunk. Then, when we move to the second chunk, we do not just pass it alone to the prompt and LLM. Instead, we also take the previous summarization result as a reference along with this second chunk. Refine essentially updates a rolling summary by iterating over the document sequentially. Each new chunk is summarized in context of the previous summaries.

For example, chunk one produces summary one. Chunk two is then combined with summary one and passed through the prompt template and LLM to generate summary two. This process continues iteratively for all subsequent chunks. Each chunk’s summary is refined with respect to all previous summaries, rolling over the results to create a more cohesive final summary. After processing all chunks, we get the final refined summary. This approach differentiates refine from MapReduce and stuff chain, as it focuses on incrementally improving the summary rather than processing each chunk independently or combining everything at once.

The practical implementation of refine chain is simple using LangChain. We use "from langchain.chains import load_summarize_chain" and create a chain with "chain = load_summarize_chain(llm=lm, chain_type='refine', verbose=True)". To get the final refined summary, we run "output_summary = chain.run(final_documents)" and then print it using "print(output_summary)". Here, each chunk is processed sequentially, and the summary is refined progressively by including the previous summary in the context.

At the end, refine summarization ensures that the summary evolves iteratively, improving with every chunk processed. The first chunk’s summary is combined with the second chunk, passed to the LLM, and this continues until all chunks are processed, giving a much more polished and coherent summary.

In this series of videos, we have covered all three types of summarization techniques: stuff document chain, MapReduce, and refine chain. We have also understood their theoretical intuition and practical applications. In the next video, we will develop an end-to-end project implementation for both structured and unstructured content using these summarization techniques.

**XV) Gen AI Projects- Youtube Video And Website Url Content Summarization**

**A) End To End Project Demo**

So finally we are going to create our end-to-end Gen AI project using LangChain, where we will summarize text from YouTube videos or any external website. In this video, I am going to show you a demo of what this entire Gen AI app actually does. I will go ahead and enter all the required values that you see on your screen.

On the left-hand side, we will use the Grok API key to access our open-source models. Here, you have the option to enter any URL. This URL can be a YouTube URL or a website URL. Once you enter the URL and click on summarize, the app will extract content from the URL and summarize it for you.

For example, I have a video on my YouTube channel titled “AI versus ML versus DL versus Generative AI.” I copy and paste the video URL into the app and click summarize. If the Grok API key is missing, the app prompts you to provide it. After adding the API key from my env file, I click summarize again. The app then fetches the transcript of the video and summarizes it. The model I am using is the Gamma model from Google, which is fully open source. I also tested other models like LLaMA 3 and Mistral, but the Gamma model provided the best results.

The summary generated explains the concepts of generative AI, which is a subset of deep learning focused on generating content. This functionality works for YouTube videos. Similarly, we can summarize content from websites. For example, I can enter a URL like docs.blacksmith.com, click summarize, and the app will fetch and summarize the content of that webpage.

I can even try other websites, like LangChain’s documentation. By copying the introduction section of LangChain.com, pasting it into the app, and clicking summarize, the app provides a concise summary of the lecture and framework content. It handles multiple sources efficiently and delivers a clear summary every time.

This demo gives an overview of the Gen AI app’s capabilities. In the next video, we will start coding step by step to implement each part of this project, making sure everything works as expected.

**B) Implementing Youtube Video And Website Url Content Summarization GEN AI App**

So finally, I’m excited to implement this amazing end-to-end project related to text summarization. I’ll start by creating an app.py file inside our project folder, which is where we’ll write the entire code. The reason for this folder structure is to keep all common libraries and resources in one place, which is useful for building robust, end-to-end projects.

Step 1: Install Required Libraries

We’ll be using a few external libraries:

validators – to validate URLs.

youtube-transcript-api – to extract transcripts from YouTube videos.

Make sure your virtual environment is active and run:

pip install -r requirements.txt


This will install all necessary libraries, including validators and youtube-transcript-api.

Step 2: Import Libraries

We start the code by importing all required libraries:

import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_grok import ChatGrok
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YouTubeLoader, UnstructuredURLLoader

Step 3: Streamlit App Setup

We set up the page, title, and subheader. Then, we create sidebar input fields for the Grok API key and the URL to summarize:

st.set_page_config(page_title="GenAI Text Summarizer")
st.title("GenAI Text Summarizer")
st.subheader("Summarize YouTube videos or websites")

with st.sidebar:
    grok_api_key = st.text_input("Grok API Key", "", type="password")
    url = st.text_input("Enter YouTube or Website URL", "", placeholder="https://")

Step 4: Validate Inputs

We check that both the API key and URL are provided and that the URL is valid:

if not grok_api_key.strip() or not url.strip():
    st.error("Please provide both API key and URL to proceed.")
elif not validators.url(url):
    st.error("Please enter a valid URL (YouTube or Website).")
else:
    try:
        with st.spinner("Processing..."):
            # Load data based on URL type
            if "youtube.com" in url:
                loader = YouTubeLoader.from_youtube_url(url, add_video_info=True)
            else:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36"
                }
                loader = UnstructuredURLLoader(urls=[url], verify_ssl=False, headers=headers)
            docs = loader.load()

Step 5: Initialize LLM and Prompt Template

We initialize the Grok model and define the prompt template for summarization:

llm = ChatGrok(api_key=grok_api_key, model="gamma")
prompt_template = "Provide a summary of the following content in 300 words: {text}"
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

Step 6: Create Summarization Chain

We define the summarization chain and generate the summary:

chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
output_summary = chain.run(docs)
st.success("Summary generated successfully!")
st.write(output_summary)

Step 7: Exception Handling

We handle any exceptions gracefully:
    except Exception as e:
        st.exception(f"Error: {e}")

Step 8: Run the App

Finally, run the Streamlit app in your terminal:

streamlit run app.py


Make sure all required dependencies (unstructured, youtube-transcript-api, validators, etc.) are installed in your environment. Test it with both YouTube URLs and website URLs to ensure it works correctly.

This completes the setup for our end-to-end text summarization app using LangChain and Grok API.

The app extracts content from the URL, generates a summary using the LLM, and displays it in Streamlit.

This is a fully functional demo ready for further enhancements.

# **XVI) Text To Math Problem Solver Using Google Gemma 2**

**A) Demo of the End to End Project**

Hello guys. So we are going to continue our discussion with respect to our LangChain series. In this video and the upcoming series of videos, we are going to discuss another amazing new project. The project is called Innovative Math and Data Source Assistant using Google Gamma 2. Google Gamma 2 is an open-source model from Google, which is quite powerful with respect to language modeling, and it gives amazing results. That is the reason we are going to use it for this project.

The main idea behind this project is that whatever question you ask this GPT-like generative AI application, related to any math problem, it should be able to give you the answer. The question can be in the form of text, and the system should automatically understand your query and provide the answer.

Additionally, when studying math, there might be extra information you want to know, such as formulas. For example, the formula for the area of a circle is A = π * r ** 2. This type of information can be pulled from external data sources, so we are integrating Wikipedia to provide relevant context along with the answer.

Let me show you a demo of this project. First, we enter the Grok API key:

grok_api_key = "your_grok_api_key_here"


Once entered, the assistant will say, “Hey, I’m a chatterbot who can answer all your math questions.” For example, if we input the following text question:

question = """
I have five bananas and seven grapes. I ate two bananas and gave away three grapes.
Then I buy a dozen apples and two packs of blueberries. Each pack of blueberries contains 25 berries.
How many total pieces of fruit do I have?
"""


The system will interpret the question, perform calculations, and generate a step-by-step answer. When we click Get Answer, it processes the numbers and provides the solution. In this case, the answer is:

answer = 69


Breaking down the calculation:

Step-by-step
bananas_grapes = 5 + 7             # 12
after_eating_giving = 12 - 2 - 3   # 7
total_apples = 7 + 12               # 19
total_blueberries = 2 * 25          # 50
total_fruit = 19 + 50               # 69


As you can see, the system handles the arithmetic automatically. It calculates all intermediate steps to arrive at the final answer. This means that you can input any text-based math problem, and the assistant will solve it and explain the steps clearly.

In the next video, we will build this application completely from scratch. We will first develop the left panel of the Streamlit app, and then build the full interface. We will also see exactly how we use Google Gamma 2 to solve math problems efficiently.

So yes, this was it from my side for this demo and theoretical overview. I will see you all in the next video where we implement everything practically from scratch. Thank you.

**B) End To End Text to Math Problem Solver Using Google Gemma2 Model Implementation**

Hello guys. So let’s go ahead and build our end-to-end generative AI application based on the demo that we have already seen. To start with, we will require the Streamlit library, as we are going to create the whole application using Streamlit. Along with that, we will import ChatGrok, because we are going to use the Google Gamma 2 model.

Additionally, we will use two important chains: LM Chain and LM Math Chain, since we need to perform mathematical calculations within the app. These libraries will be useful for our math-solving functionality. Furthermore, to enable interaction between the agents and access external information, we will also use Wikipedia, PromptTemplate, AgentType, initialize_agent, and Tools from LangChain. If you want to load the API key from an environment file, you could use load_dotenv, but in this case, we will provide the API key directly from the website interface. Finally, for tracking interactions, we will use the StreamlitCallbackHandler.

First, let’s set up our Streamlit app. We create our Streamlit script and configure the page using:

st.set_page_config(
    page_title="Text to Math Problem Solver and Data Search Assistant",
    page_icon="🧮"
)
st.title("Text to Math Problem Solver using Google Gamma 2")
st.write("This app helps solve math problems and fetch relevant data from Wikipedia.")


Next, we will create a sidebar input for the Grok API key:

grok_api_key = st.sidebar.text_input("Enter your Grok API key", type="password")
if not grok_api_key:
    st.info("Please add your Grok API key to continue.")
    st.stop()


Once the API key is provided, we can initialize our language model:

llm = ChatGrok(model="gamma", api_key=grok_api_key)


Now we will initialize our tools. First, we create a Wikipedia tool:

wikipedia_tool = WikipediaWrapper()
wikipedia_search = Tool(
    name="Wikipedia",
    func=wikipedia_tool.run,
    description="Tool for searching the internet and retrieving information to solve math problems."
)


Next, we initialize the math-solving tool using LM Math Chain:

math_chain = LMMathChain(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Tool for solving math expressions. Only provide mathematical expressions."
)


After setting up the tools, we define the prompt template for the agent:

prompt = """
You are an agent tasked with solving users' mathematical questions.
Provide detailed step-by-step explanations and display answers point-wise.
Question: {question}
"""
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)


We then combine the LM model and the prompt template into an LM chain, which can later be used as a reasoning tool:

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=LLMChain(llm=llm, prompt=prompt_template).run,
    description="Tool for answering logic-based and reasoning questions."
)


With our tools ready, we initialize the agent by combining all tools:

assistant_agent = initialize_agent(
    tools=[wikipedia_search, calculator_tool, reasoning_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


We also set up Streamlit session state to maintain conversation history:

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a math chatbot who can answer all your math questions."}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


Next, we create a function to generate responses from the agent:

def generate_response(user_question):
    response = assistant_agent.invoke({"input": user_question})
    return response


Now we set up the user interface to input a question and start the interaction:

question = st.text_area("Enter your question", "I have five bananas and seven grapes. I ate two bananas and gave away three grapes. Then I buy a dozen apples and two packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            cb = st.container()
            response = assistant_agent.run(input=question, callbacks=[cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            st.success(response)
    else:
        st.warning("Please enter your question.")


Finally, after setting up the API key, initializing the LLM, creating tools, chaining the prompt, and building the session state, our app is ready to run. You can launch it using:

streamlit run app.py


When the app is running, you can input any text-based math problem, and the agent will generate a step-by-step response. For example, it can calculate the number of fruits, solve logical reasoning with apples, or answer complex arithmetic questions using the reasoning tool.

This completes the Text-to-Math Problem Solver using Google Gamma 2, Wikipedia integration, and LangChain tools. You can extend this project by adding more tools, better prompts, or multi-step problem-solving capabilities.

# **XVII) Huggingface And Langchain Integration**

**A) Introduction To Huggingface And Langchain Integration**

Today, we are going to continue our discussion on LangChain and start exploring Hugging Face. In our previous videos, we have already seen how to use Hugging Face embeddings, but in this module, we will dive deeper into Hugging Face’s integration with LangChain.

On May 14th, 2024, Hugging Face published an article announcing a new partnership with LangChain, making it easier to work with Hugging Face language models directly within the LangChain ecosystem. Before this integration, developers had to use different Hugging Face libraries to call models, which was a bit cumbersome. With the new integration, this process is now simplified, and since our course focuses on the LangChain ecosystem, we will leverage this integration for our projects.

First, you need to create an account on Hugging Face
 if you don’t already have one. Hugging Face hosts a huge repository of models, not just one or two. You can find open-source models as well as paid models. For example, the Google Gamma 2 9B parameter model is available, along with many other models for NLP, computer vision, and multimodal tasks.

Paid models require setting up API endpoints and may require a credit card, but Hugging Face still provides a wide range of free open-source models that can be used without much hassle. The platform categorizes models by tasks such as text-to-text, image-to-text, multi-modal models, visual question answering, document question answering, and more.

For instance, if you want to perform image-text-to-text tasks, Hugging Face has models that handle both images and text. You can find the model details, requirements, and sample code to call these models either via the transformers library or via Hugging Face endpoints.

However, our main aim is to simplify the calls to Hugging Face language models using LangChain. This allows us to integrate Hugging Face models directly into LangChain workflows without dealing with low-level API calls or multiple libraries.

In addition to multimodal models, Hugging Face provides NLP-specific models, including text classification, question answering, and summarization. For example, you can check the Gamma 2 9B model for text summarization, test the inference API, and observe how large models require special handling due to their size. Smaller models, such as Meta LLaMA 8B, can be tested directly via the Hugging Face inference API.

The Hugging Face documentation provides detailed guidance on installation, model usage, embeddings, and API access. With the integration into LangChain, we now have a Python package that brings the power of Hugging Face models into the LangChain ecosystem, making it easier to implement end-to-end generative AI applications.

In the next video, we will cover practical implementations with Hugging Face, including embeddings and using models for real applications. After that, we will finally build a complete end-to-end project leveraging both LangChain and Hugging Face.

So, I encourage you to explore Hugging Face, check out different models, and read their documentation. This will give you a good foundation before we dive into the hands-on examples in the upcoming sessions.

That’s it from my side for this video. I’ll see you all in the next video. Thank you!

**B) Langchain And Huggingface Integration Practical Implementation**

Today we are going to continue our discussion on Hugging Face and LangChain. As mentioned earlier, Hugging Face and LangChain have now partnered together and created a new package called langchain-huggingface, which simplifies working with Hugging Face models in the LangChain ecosystem.

Step 1: Installation

First, we need to install the required package:

pip install langchain-huggingface


In my setup, I have a folder called ninth_folder with an experiment.ipynb file. We are using Python 3.10 in the virtual environment.

You can also install from requirements.txt:

pip install -r requirements.txt


Next, we need the Hugging Face Hub, which helps with API calls to Hugging Face models:

pip install huggingface_hub


This package will allow us to interact with Hugging Face APIs directly, which is useful for both free and paid models.

Step 2: Setting Up Environment Variables

We will store our Hugging Face API token in a .env file and load it using Python:

import os
from dotenv import load_dotenv

load_dotenv()  # loads environment variables from .env
HF_TOKEN = os.getenv("HF_TOKEN")


This token is necessary to authenticate API calls for accessing Hugging Face models.

Step 3: Using Hugging Face Models

There are two main ways to access Hugging Face models:

Hugging Face Endpoints (serverless API, recommended for free or Pro users)

Direct model access with repo_id

For this tutorial, we will use Hugging Face Endpoint from langchain-huggingface:

from langchain_huggingface import HuggingFaceEndpoint

repo_id = "mistralai/Mistral-7B-Instruct"
lm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=150,
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN
)


Now you can invoke the model like this:

response = lm.invoke("What is machine learning?")
print(response)


You can also try other questions:

response = lm.invoke("What is generative AI?")
print(response)

Step 4: Using Other Models

You can access multiple models available on Hugging Face. For example, the Google Gamma 2 model:

repo_id = "google/gamma-2-7b"
lm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=150,
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN
)
response = lm.invoke("What is machine learning?")
print(response)


Note: Some large models may not load automatically, and in such cases, you need a dedicated endpoint, which requires a paid Hugging Face account.

Step 5: Creating Prompt Templates and Chains

LangChain allows you to use prompt templates and LM chains for structured question-answering:

from langchain import PromptTemplate, LLMChain

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

chain = LLMChain(llm=lm, prompt=prompt)
response = chain.run("Who won the Cricket World Cup 2011?")
print(response)


Output:

The 2011 Cricket World Cup was held in the Indian subcontinent, and the winner was India who defeated Sri Lanka by six wickets.

Step 6: Hugging Face Embeddings

Hugging Face also provides open-source embeddings, which can be used with LangChain:

from langchain.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding = embed_model.embed_query("This is a sample text")
print(embedding)


These embeddings can be used in retrieval-based applications or RAG pipelines.

Step 7: Hugging Face Endpoints (Paid & Free)

Free users: Limited serverless API requests.

Pro/Enterprise users: Can create dedicated endpoints for large models and higher usage.

Credit card required for paid endpoints.

For most local experiments, free endpoints are sufficient.

In the next video, we will build an end-to-end project using Hugging Face Endpoints and LangChain.

**C) End to End Gen AI Project With Langchain And Huggingface**

Today, we are going to see an end-to-end project using LangChain and Hugging Face integration. In this project, we will replace the previous Grok API and Gamma 2 model with a Hugging Face Endpoint model to perform text summarization.

Step 1: Project Setup

We are using a folder called ninth_huggingface_langchain with a app.py file. Previously, we had a text summarization app that could summarize content from a YouTube URL or any website URL using Grok API.

Now, we will switch to Hugging Face models. First, install the required packages:

LangChain Hugging Face integration: "pip install langchain-huggingface"

Hugging Face Hub: "pip install huggingface_hub"

Also, make sure you have a .env file containing your Hugging Face API token:

HF_TOKEN=<your_huggingface_api_token_here>


We will load it in Python like this:

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

Step 2: Update API Key in App

In your app.py, replace the Grok API key with your Hugging Face API token. For example:

Replace this
grok_api_key = os.getenv("GROK_API_KEY")

With Hugging Face API token
hf_api_token = os.getenv("HF_TOKEN")

Step 3: Import Hugging Face Endpoint

Next, import the required Hugging Face library from LangChain:

from langchain_huggingface import HuggingFaceEndpoint


Now, define the model using the HuggingFaceEndpoint class. For example, using Mistral-7B-Instruct:

repo_id = "mistralai/Mistral-7B-Instruct"

lm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=150,
    temperature=0.7,
    huggingfacehub_api_token=hf_api_token
)

Step 4: Integrate with Streamlit

Update the Streamlit input to ask for the Hugging Face API token:

import streamlit as st

hf_api_token = st.text_input("Enter your Hugging Face API token")


Comment out any previous Grok API code and replace it with the Hugging Face endpoint model:

lm = GrokAPI(model="gamma-7B", api_key=grok_api_key)
replaced with
lm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=150,
    temperature=0.7,
    huggingfacehub_api_token=hf_api_token
)

Step 5: Define Summarization Function

We can create a helper function to summarize content from a URL:

def summarize_content(url):
    Extract text from URL (YouTube or website)
    content = extract_text_from_url(url)  # Assume this function exists
    prompt = f"Summarize the following content:\n\n{content}"
    summary = lm.invoke(prompt)
    return summary

Step 6: Run the App

Finally, run the Streamlit app:

streamlit run app.py


Now, you can enter your Hugging Face API token and provide a YouTube or website URL. The app will summarize the content using the Hugging Face model.

For example, summarizing the video "AI vs ML vs Generative AI":

url = "https://www.youtube.com/watch?v=<video_id>"
summary = summarize_content(url)
print(summary)


Similarly, summarizing a webpage:

url = "https://docs.langchain.com/"
summary = summarize_content(url)
print(summary)


Note: The output may differ slightly from the Google Gamma 2 model since the Hugging Face model may be smaller, but it still provides a concise and useful summary.

Step 7: Explore and Extend

Now that your app works with Hugging Face, you can:

Try other models from Hugging Face Hub

Integrate embeddings using "from langchain.embeddings import HuggingFaceEmbeddings" for vector-based tasks

Build RAG pipelines or Q&A apps with your endpoint

Example embedding usage:

from langchain.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding = embed_model.embed_query("This is a sample text")
print(embedding)

This gives you a fully functional end-to-end project using LangChain + Hugging Face, ready to extend for summarization, Q&A, or retrieval tasks.

# **XVIII) Pdf Query RAG With Langchain And AstraDB**

**A) End To End Project With PDf Query RAG With Langchain And AstraDB**

So guys, in this video we are going to create an amazing LLM project, which is a PDF Query Application using LangChain and Cassandra DB. Cassandra DB will be created on a platform called DataStax, which allows you to create Cassandra DB in the cloud and perform vector search. Vector search is essential when working with large documents or building Q&A applications from PDFs.

Before we dive into coding, let’s understand the architecture. Initially, you have a PDF of any size or number of pages. First, we will read the document using LangChain, which has functionalities to handle tasks like this efficiently. After reading the document, we will split the PDF content into text chunks. These chunks are created based on a specific token size. Here’s an example of reading the document and splitting it into chunks:

from PyPDF2 import PdfReader

pdf_reader = PdfReader("budget_speech.pdf")
raw_text = ""
for i, page in enumerate(pdf_reader.pages):
    raw_text += page.extract_text()


Next, we will convert these chunks into text embeddings using OpenAI embeddings. Text embeddings convert text into vectors so we can perform tasks like similarity search. OpenAI embeddings are used because they handle this efficiently:

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_API_KEY")


Once we have embeddings, we need to store them in a database since large PDFs will generate many vectors. For this, we use Cassandra DB (or Astra DB, which is Cassandra hosted on DataStax). Cassandra is a NoSQL database designed for handling massive amounts of data with scalability and high availability. To connect and initialize the database, we use:

import cashew

cashew.init(token="YOUR_ASTRA_TOKEN", db_id="YOUR_ASTRA_DB_ID")


Now, we create the LangChain vector store using Cassandra. This will wrap all vectors in a convenient structure and push the data to the database while automatically applying embeddings:

from langchain.vectorstores.cassandra import Cassandra
from langchain.vectorstores import VectorStoreIndexWrapper

vector_store = Cassandra(
    embeddings=embeddings,
    table_name="qa_mini_demo",
    keyspace="langchain_db"
)


Next, we split the PDF text into chunks using a character text splitter so that the token size does not increase too much. We also set chunk overlap for better context:

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200
)
chunks = text_splitter.split_text(raw_text)


We then add these chunks to the Cassandra vector store, which applies embeddings as it inserts them:

vector_store.add_texts(chunks[:50])  # Insert top 50 chunks for testing


Finally, we can query the PDF. When a human provides a text query, embeddings are generated and a similarity search is performed on the Cassandra DB. Here’s a sample interactive loop:

while True:
    query = input("Enter your question (or type 'quit' to exit): ")
    if query.lower() == "quit":
        break
    results = vector_store.similarity_search(query, k=4)
    for result in results:
        print(result.page_content)


For example, asking "How much is the agriculture credit target?" may return:
"Agriculture credit target will be increased to 20 lakh crore with the focus on animal husbandry, dairy, and fisheries."

Similarly, you can query GDP, budgets, or any other topic in the PDF.

This approach allows you to build scalable Q&A applications for large PDFs using LangChain and Cassandra DB, powered by vector search. DataStax Astra makes this process smooth and provides a free vector search-enabled Cassandra DB for your projects.

**XIX) MultiLanguage Code Assistant Using CodeLama**

**A) End To End MultiLanguage Code Assistant Implementation**

In this video, we are going to create our own code assistant, an end-to-end project using Code Llama. So, what exactly is Code Llama? It is an open-source large language model (LLM) developed by Meta, which allows you to build a personal code assistant capable of generating code from custom prompts. We will go step by step to implement this. I’ll also be using Ulama to access the model locally and in the cloud.

Code Llama is a state-of-the-art LLM specialized for coding tasks. For example, if you provide a prompt like "Python code for Fibonacci series", it will return the complete Python code. As of January 29th, 2024, the latest update includes three main variants: Code Llama 70B Foundation, Python 70B (specialized for Python), and Code Llama 70B Instruct, which is fine-tuned for understanding natural language instructions. Essentially, Code Llama can generate code and natural language explanations from both code and text prompts. It’s free for research and commercial use and is based on Llama 2, fine-tuned for coding. Code Llama supports multiple languages including Python, C++, Java, PHP, TypeScript, C#, and Bash.

To get started, you first download Ulama on Windows. After downloading, click the .exe file to install it. Ulama runs in the background, usually on a localhost IP. In the Ulama models directory, you will find Code Llama. You can run it via the command prompt using "llama run Code Llama". For example, if you provide a prompt like "Provide Python code to perform binary search", Code Llama will return the complete code instantly.

Next, we create our custom GPT-like application called Code Guru using Code Llama. Open VS Code and create a virtual environment. In your requirements.txt, install "langchain" and "gradio". Then, create a model file where we define the behavior of our assistant:

from code_llama import CodeLlama

temperature = 1

system_prompt = """
You are a code teaching assistant named Code Guru.
Created by Krish.
Answer all code-related questions.
"""


Here, we set the temperature to 1 to make the model more creative and define the system prompt describing the assistant’s role.

To run the model using Ulama, navigate to the folder containing your model file and execute:

ulama create -f model_file.py CodeGuru


Replace model_file.py with the actual model file name. Once running, you can test your assistant with:

llama run CodeGuru


and ask questions like "Who are you?" or "Who created you?". The assistant will respond with: "I'm Code Guru, an AI system designed to help with code queries. Created by Krish."

Now, we integrate this with a Gradio front end. First, import the required libraries:

import requests
import json
import gradio as gr


We define the URL and headers for the API:

url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}


Next, we create a function to generate responses from Code Guru:

history = []

def generate_response(prompt):
    history.append(prompt)
    final_prompt = "\n".join(history)
    data = {
        "model": "CodeGuru",
        "prompt": final_prompt,
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        output = json.loads(response.text)["response"]
        return output
    else:
        return f"Error: {response.text}"


This function appends the prompt to a history, sends the request to Code Guru via the API, and returns the generated response.

For the Gradio interface, we use a text box for input and another for output:

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your prompt"),
    outputs="text"
)

interface.launch()


Once launched, you can open the Gradio UI and ask questions like "Provide Python code to perform binary search" or "Provide Java code". The assistant responds instantly with the correct code. It also keeps a history of previous prompts, allowing a continuous conversation with context. For example, asking "Provide Python code to create Adam optimizer" will return a complete explanation along with the Python implementation. Similarly, asking "Provide Python code for Fibonacci series" will give the expected result.

This setup demonstrates how you can create a multi-language code assistant using Code Llama and Ulama. With this, you have an end-to-end solution that combines a local LLM with a user-friendly Gradio interface, capable of generating accurate code responses across different programming languages.

# **XX) Deployment Of Gen AI Apps In Streamlit and Huggingspace**

**A) Deployment OF Gen AI APP In Streamlit Cloud**

Now we have developed so many different applications. I will just try to show you how you can go ahead and deploy these kinds of applications. From all these examples, you can take up any example that you want. But I will go ahead and take this particular search engine which I had actually developed. I will go into its file. When I go over here, you'll be able to see all these files I have actually arranged in this way. This is my V and V folder and this is my requirement.txt.

To start the work over here, what I am actually going to do is take this requirement.txt and copy it inside this particular folder. This is my search engine right now. Let me just go ahead and deploy this entire application as a Streamlit web app or on Streamlit Cloud. Since this is a Streamlit web app, I will first go ahead and show you the deployment with Streamlit Cloud. I will just go ahead and type "Streamlit Cloud" and hit enter.

Here you'll be able to see that Cloud Streamlit provides a faster way to build and share data apps. But before that, let me go to my GitHub account. Inside this particular GitHub account, I will create a new repository. I will click on "New Repository" and write my repository name as "Search Engine LM". I will add a README file and a license, like the "General Public License", but you can select any license you like. Then I will click "Create repository".

One thing you should understand is the reason why I'm using Streamlit Cloud—it allows you to deploy any number of apps over here. Now, inside this repository, I will open my folder and select all these files. I will drag and drop them into the repository. I don’t even require the Jupyter notebook .ipynb file because I just need "app.py" and "requirement.txt". Once done, I will commit the changes. Now I have my entire project on GitHub.

Next, let's go to Streamlit Cloud. I will sign in here. You can deploy any kind of Streamlit web app this way, whether it is a generative AI application or a machine learning application. Here, you can see some examples that I have already uploaded. Streamlit Cloud is very convenient because you don’t have to pay anything for deployment, and it provides a URL that ends with streamlit.io. I will click "Create app".

It will ask if I already have an app. Here, it has an option to integrate with GitHub. Since I have uploaded the app on GitHub, I will click this option. It will first ask for authentication with GitHub, so please ensure you authenticate with your GitHub account. After authentication, I will search for my repository, "KrishNayak06/SearchEngineLM" in the main branch. My main file is "app.py", so I will select "app.py". Please make sure you have "requirement.txt" in the repository; otherwise, Streamlit won’t know which libraries to install.

Here, you can see the domain for your deployed app. You can also add secret keys, like API keys, as key-value pairs. For example, you can add an OpenAI API key using "st.secrets". In your code, you would access it like this:
"openai_api_key = st.secrets['OPENAI_API_KEY']"
This allows you to securely use keys in your app. In my case, I am not using any secret keys in the deployment because I will provide the Hugging Face API token from the frontend.

Once everything is set, I will deploy the app. Streamlit will first install all the dependencies listed in "requirement.txt". After installation, your app will be live. In my app, I copy and paste my API key in the frontend input and test queries like "Tell me about machine learning". The app uses agents to search different sources like Wikipedia and arXiv to provide responses. For example, asking "Attention is All You Need research paper" communicates with arXiv and returns relevant information. Similarly, asking "Tell me about generative AI" fetches content from multiple sources.

This deployment demonstrates how easy it is to get a working application online using Streamlit Cloud. Another platform for deployment is Hugging Face Spaces. Hugging Face provides a server space where you can deploy applications with a URL. You can discover different Spaces, upload your app, and it will be live just like Streamlit.

In the next video, I will show how to deploy your application using Hugging Face Spaces. I hope you found this tutorial on deployment useful. Thank you, take care, and have a great day.

**B)  Deployment Of Gen AI App In Huggingface spaces**

So we are going to continue our deployment series. In our previous video, I have already shown you how to deploy your generative AI application using a Streamlit web app with the help of Streamlit Cloud. The steps were quite easy—we just needed to put all the code in our GitHub repository and connect it with Streamlit Cloud (or Streamlit Lab).

Now, I am going to deploy this entire code solution in Hugging Face Spaces. Spaces allow you to deploy generative AI applications or any machine learning applications. Step by step, I will show you how to do this in an easy way. If you browse Spaces, you'll find many use cases that are already deployed, and you can even check out the code for reference.

There are several steps involved in deploying a solution in Hugging Face Spaces. Here, we will also use something called "GitHub Actions". If you search for "Hugging Face Spaces GitHub Action", you'll see a repository titled "Managing Spaces with GitHub Action". This repository contains the code that we will use. Essentially, we will create a "YAML" file that tells GitHub Actions how to push our code to the Hugging Face Space for deployment.

First, I will create a new folder inside the GitHub repository called ".github". When creating a CI/CD pipeline, you usually need a "YAML" file that contains all the configurations for pushing code to deployment servers. Inside the ".github" folder, I will create a subfolder "workflows", and within this, a file named "main.yaml". This YAML file contains the instructions that tell GitHub Actions to push the repository to Hugging Face Spaces whenever a commit occurs.

The code in the YAML file contains key-value pairs specifying the workflow. It handles syncing to the Hugging Face Hub, pushing from the "main" branch, running the job on "ubuntu-latest", checking out the code, and finally pushing it to the hub using a secret token called "HF_TOKEN". I will show you how to generate this token and where to use it.

Next, let's go to Hugging Face Spaces and create a new space. I will name it "search_engine_LM". I will select a license, such as "Apache 2.0". Although this is a Streamlit app, Hugging Face also supports static templates and Docker-based deployments. The default hardware is free CPU with "2 vCPUs" and "16 GB RAM", which is sufficient for most applications. I will make this space public and click "Create Space".

Once the space is created, you can clone it and start working. Currently, there are no files in this space, so we need to push the files from our GitHub repository to this Hugging Face Space.

To generate the "HF_TOKEN", go to your Hugging Face account, click "Settings", then "Access Tokens", and create a new token. Name it "token1" and select the type "Write" because we need write access to push files into the Hugging Face Space. Copy this token and save it securely.

Next, I will set the Hugging Face username and space information in the YAML workflow file. My username is "Krishna06" and the space name is "search_engine_LM". The workflow pushes the "main" branch of our repository to this space using the "HF_TOKEN" secret.

Inside the Hugging Face Space settings, under "Secrets and Variables" → "Actions", I will create a new repository secret named "HF_TOKEN". This secret allows the YAML workflow to authenticate and push files to the space. Once the secret is added, the CI/CD pipeline is ready.

After committing changes in GitHub, the GitHub Actions workflow starts automatically. The workflow may show errors initially if previous test pushes were made to the same space. To fix this, we can update files and commit them again. For example, I edited a file and committed the changes. GitHub Actions then triggered the push to Hugging Face Spaces.

Sometimes the README file can cause build errors due to special characters. I edited the README file to include a short description, like "This is my search engine" and "Search Engine with LM". After updating, committing, and pushing, the build starts. Hugging Face installs all dependencies from "requirements.txt" and builds the container.

Once the build completes, the application runs on Hugging Face Spaces. You can share the URL to demonstrate your deployed app. In my app, I update my environment variable and Grok API key, then search for queries like "What is machine learning?". The app fetches information from DuckDuckGo search, Wikipedia, or other sources. Similarly, asking "What is generative AI?" provides relevant responses.

In summary, this deployment mechanism demonstrates how to deploy generative AI applications using Hugging Face Spaces, GitHub workflows, and GitHub Actions. The CI/CD pipeline automates pushing updates from your repository to Hugging Face Spaces. This is an easy and scalable method for deploying any AI or machine learning application.

# **XXI) Generative AI with AWS (Bonus)**

**A) Life Cycle Of Gen AI Project In AWS Cloud**

So guys, we are going to continue the generative AI on cloud series. And in this video I'm going to probably discuss about the gen AI project lifecycle. Now, since you already know that, we are definitely going to develop a lot of applications specifically in cloud from the basic data ingestion till the deployment. So it is very much necessary that you actually understand a generic workflow of the gen AI project lifecycle.

So let me quickly go ahead and let me go ahead and write some amazing things for you in this particular notebook, and I will be explaining you completely, step by step, how you can probably see or follow a project life cycle. And as we go ahead, there will be a lot many things that will be coming, like LM ops platform. And we will be working specifically with Azure AI studio, AWS SageMaker studio and all. So definitely both the clouds will get covered.

So before I go ahead, please make sure that you keep the like target of all this kind of videos till 1000. That will definitely motivate me. And I've been exploring many more things so that you get the right kind of guidance and knowledge. So let me go ahead and let me start the gen AI project lifecycle.

With respect to this gen AI project life cycle, I would like to make this entire life cycle into 4 to 5 steps. The first step is basically defining the use case. So what kind of use case are you solving? Then this use case can be a RAG application, can be a text summarization application, can be a chatbot. So based on different use cases that actually depends on your requirements, your company requirements. So this is the first step. You really need to define the use case that you're specifically doing.

Now with respect to this particular use case we usually take this entire module into the scope part. So this is basically my scope, right, if I basically use a generic term. Now once you define a use case, let’s say that I am going to probably develop a RAG application. In that, I'm going to definitely use vector databases. I may have a lot of PDF files. I also need to probably convert that into vectors and store it in some kind of vector store DB. So some kind of use case you really need to define and all the requirements that is required in that particular use case.

Coming to the next step, which is super important because this step will be involving two important things, and that is nothing but choosing the right model. When I say choosing the right model, here there are two different things that you can probably split this into. One, whether you are using some kind of foundation models. So here I'm going to write whether you are using a foundation model and solving a use case. This is the one category that I would like to divide this particular module into.

The other category is that whether you want to build your own custom LLM. Custom LLM is nothing but building your LLM from scratch. Now, see, there are two things over here. When I say foundation model, foundation models are already those larger models like OpenAI, Llama 2, Llama 3, Google Gemini Pro. So these are all very huge foundation models. And for most of the generic use cases, you can directly use those kind of foundation models and you can solve the use case itself.

Now with respect to these foundation models, we can also further go ahead and do fine-tuning. Let’s say I have a foundation model which I am specifically using to solve my business use cases. On top of that, if I really want to make this foundation model behave well for my own custom data, then what I can do on top of this foundation model is I can use LoRA techniques and I can probably fine-tune all this kind of models.

So this is one of the steps. The second step that I have written over here is custom LLM. Custom LLM is nothing but building your LLM from scratch. And obviously, there is a lot of benefit if a company is building an LLM model completely from scratch for its specific use cases, but a lot of resources will definitely be required. We have to really take care of model hallucination, many things and all as we go ahead. But yes, I've also seen many, many companies developing their own custom LLM model.

So choosing the right model, or what kind of models you're specifically using to solve this particular use case, becomes the second important module with respect to this gen AI project life cycle. And obviously I've spoken about foundation models both in AWS, in Google, in Microsoft Azure. Currently Microsoft Azure AI Studio specifically has all the access of OpenAI services, obviously because it is investing a huge amount of money over there.

Now once you select the right kind of model, there are three main tasks that you probably do going forward. The first task is nothing but you can specifically use prompt engineering and solve a use case. The second task that you can actually do is fine-tuning. So with the help of fine-tuning, also, you can probably develop your own custom LLM model. And on top of that, you can basically do it.

Let’s say you're completely creating your LLM model from scratch. One more important mechanism that you have is nothing but aligning, or you can probably say training with human feedback. Training with human feedback is one of the very important steps that is actually used while you are training your LLM models. How an LLM model is basically trained. I've already created a video in my playlist with respect to LangChain and all Generative AI playlist, you can probably go head over there. Fine-tuning, how to specifically do fine-tuning and all that, also I've actually shown you.

The reason why I'm showing you this generative AI project life cycle is because tomorrow when I'm probably creating videos, in the upcoming videos related to this series, over there you'll be seeing all these particular steps going ahead.

Now, once you do all the steps, the further step is something called evaluation. Evaluation is basically seeing how your model is performing by performing all these particular steps. There are also different performance metrics which we are probably going to follow. These two steps I would like to combine and say something like this — adapt and align models. So this will be the specific model approach that we specifically use for this purpose.

Now over here, your model will be ready, everything is perfect, or you are able to solve the use cases. Let’s say your performance metrics is increasing and it is saying that now your model is ready. Now it comes to the deployment part. With respect to the deployment part, I would definitely say deployment, and further you also need to do a lot of integration with different applications. So I will probably say application integration.

And here, what we do is we specifically perform two major steps. One, we optimize and deploy models. And this deployment is specifically done for inferencing. Here is where most of your cloud platforms and here is where your LM Ops is used. Different inferencing techniques are there. One technique I've already covered with respect to a platform, which is called Grok. It uses an inferencing technique which is called LP. So it is always a good idea that you should definitely know multiple ways of inferencing.

See, at the end of the day, whatever models you create, unless and until the inferencing is not fast, you definitely cannot use those things. So it is very much necessary that you know the idea of this module extensively, because tomorrow building all these things is very easy. Fine-tuning is very easy, you definitely have a template, a framework, a dataset preparation, and all, and you can perform this particular step.

That is the reason in this series of videos you'll be seeing how much I will be focusing on LM Ops platforms. And I will also show you multiple platforms which can definitely make your inferencing very good. This is the most important thing here. Definitely, we'll be using AWS and Azure. You can also use GCP. And we'll see what all services they have specifically provided for the inferencing purpose again. But initially, our focus will definitely be on AWS.

Then the second step after we do the deployment in the application integration is that we build LLM-powered applications. Because your integration is done, your API is created, now it’s all about how well you can actually build the solutions. You can solve different use cases and all.

So this overall gives a brief idea about the entire AI project life cycle. Since we have already started this journey on cloud, this is necessary to know and you should probably follow all the steps. And whenever I create any videos with respect to AI on AWS, all these steps will be considered in mind and it will be shown to you.

**B) Introduction To AWS Bedrock With Implementation**

Guys, here is an amazing crash course on Amazon Bedrock. Many people were requesting this. So, what exactly is Amazon Bedrock? It is the easiest way to build and scale generative AI applications within your AWS platform. Many companies are using this because there are so many providers of LLMs for different text generation and image generation tasks. There is OpenAI, Anthropic’s Claude, Google, Amazon’s own LLM called Titan, and Meta’s LLaMA 2. The main problem currently is that every API and model has a different setup. Amazon Bedrock provides one AWS platform where all the models are available, and through API calls, you can use any of them. OpenAI isn’t available yet on Bedrock, but almost all other major models are.

The importance of Bedrock is that you don’t have to worry about scalability or deployment. While the cost is slightly more than OpenAI currently, it might reduce in the future. Amazon Bedrock is a fully managed service that makes foundation models from leading AI startups and Amazon available via an API. You can choose from a wide range of models to find the one that suits your use case. With Bedrock’s serverless experience, you can quickly start, privately customize a foundation model with your own data, and integrate it into your application using AWS tools. You don’t need to worry about deployment, scaling, or server management.

Different models supported include AI21 Labs’ Jurassic-2 series, Amazon Titan models, Anthropic Claude models, Meta LLaMA 2, Stability AI’s Stable Diffusion, and Cohere’s Command models. You can perform tasks like chat, text generation, image generation, code generation, content creation, and contract entity extraction. Bedrock also provides hands-on labs and basic learning courses to get started.

For example, you can use Titan Text G1 for meeting transcript action items or advanced Q&A with citations, LLaMA 2 Chat 13B for chain-of-thought reasoning, or Stable Diffusion for image generation. Bedrock provides an interactive playground to test prompts, such as generating an HD image of a beach at sunset.

To get started with Python, first create a new environment:

conda create -p venv python=3.10 -y
conda activate venv
pip install -r requirements.txt


The requirements include libraries like boto3 and awscli. Then, create an IAM user in AWS, assign necessary permissions, generate an access key, and configure it using the CLI:

aws configure
AWS Access Key ID: <your_access_key>
AWS Secret Access Key: <your_secret_key>
Default region name: us-east-1
Default output format: json


Make sure the model access is granted in the US East-1 region, as some models require requesting access before usage.

For example, using LLaMA 2 for text generation in Python:

import boto3
import json

prompt_data = "Act as Shakespeare and write a poem on machine learning."

bedrock = boto3.client(service_name="bedrock-runtime")
payload = {
    "prompt": prompt_data,
    "max_gen_length": 100,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)
model_id = "<your_model_id>"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_text = json.loads(response.get("body").read())["generation"]["text"]
print(response_text)


Similarly, for content generation using Claude (Cloudy) model:

prompt_data = "You are an expert social media content generator."

payload = {
    "prompt": prompt_data,
    "max_tokens_to_sample": 512,
    "temperature": 2.8,
    "top_p": 0.8,
    "stop_sequences": ["\n"]
}

body = json.dumps(payload)
model_id = "<cloudy_model_id>"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_text = json.loads(response.get("body").read())["completions"][0]["data"]["text"]
print(response_text)


For image generation with Stable Diffusion:

prompt_data = "Provide me a 4k HD image of a beach with blue sky, rainy season, cinematic display."

payload = {
    "text_prompts": [{"text": prompt_data, "weight": 1.0}],
    "cfg_scale": 7.0,
    "height": 1024,
    "width": 1024,
    "steps": 50,
    "seed": 123
}

body = json.dumps(payload)
model_id = "stability.stable-diffusion-xl-1024-v1-0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

image_bytes = base64.b64decode(response["body"]["artifacts"][0]["base64"])
with open("output/image.png", "wb") as f:
    f.write(image_bytes)


This will generate a high-quality image in your output folder. You can experiment with different prompts, models, and parameters to create text, code, and images for multiple use cases. Amazon Bedrock provides a unified platform to manage models, scale easily, and integrate AI into your applications without worrying about deployment or server management.

**C) Document Q&A RAG With Langchain And Bedrock**

In this particular video, we are going to discuss an end-to-end Language Model (LM) project using AWS Bedrock and LangChain. This video was highly requested by many people, and the project we are going to develop is a document Q&A application. Specifically, the application will harness multiple models provided by AWS Bedrock, such as Cloudy, Llama 2, and optionally Amazon Titan. You can choose whichever models you prefer, and the goal is to implement the entire application from scratch. To illustrate, a quick demo is provided where the system functions as a Q&A tool over multiple PDFs. These PDFs are stored as vector embeddings inside a vector store, and whenever a query is asked, the system leverages the chosen Bedrock model to retrieve an answer from the documents. For example, if you ask “What are Transformers?” and select the Cloudy output, it will call the Cloudy API to fetch the response. Similarly, selecting Llama 2 fetches the answer from the Llama 2 model. This demonstrates how queries are processed against the PDF documents, and the user can get relevant responses in real time.

The system may take some time initially because the code is not fully optimized yet, but after the first run, subsequent queries will perform efficiently. For example, when asking “What is YOLO?” through the Llama 2 output, the response is retrieved directly from the API model without delay. This project builds on previous sessions where we explored the power of Cloudy API and Llama 2, demonstrating how to invoke models and retrieve answers effectively.

The first step in development involves creating a Python file, app.py, where all the code will reside. Before coding, several libraries and tools must be installed. These include PyPDF for PDF handling, LangChain for connecting with LLMs and embeddings, Streamlit for the user interface, and FAISS (or Chroma) for vector embeddings and storage. Additionally, Boto3 and AWS CLI are required to configure AWS credentials and interact with Bedrock. A virtual environment should be created, and all dependencies installed using a requirements.txt file. Full instructions and playlist references are provided for those unfamiliar with these steps.

The project consists of two main steps: data ingestion and LM model integration, with a pre-step of data ingestion. In the data ingestion step, the application reads all PDFs from a specified folder and splits them into manageable chunks using a Recursive Character Text Splitter. This ensures that large documents are processed efficiently. The chunks are then converted into vector embeddings using a model like Amazon Titan, which is called via LangChain. These embeddings are stored in a vector store (FAISS or Chroma), which allows for fast similarity search when queries are made. While Amazon Titan embeddings are used in this project, alternatives like OpenAI or Google Gemini embeddings can also be applied.

Once the vector store is ready, the second step involves querying the LM models. When a question is asked, a similarity search retrieves the most relevant chunks from the vector store. These chunks are then passed to the LLM along with a prompt template, which instructs the model to summarize or answer the query in detail (for example, in 250 words). This process allows the model to generate concise, accurate answers based on the context of the PDF documents.

The app.py file begins with importing essential Python libraries such as json, os, numpy, and AWS-related libraries like Boto3. From LangChain, BedrockEmbeddings is imported for embedding generation, and Bedrock for LM model calls. Additional imports include document loaders from pyPDF and utilities for text splitting and vector store creation. A Bedrock client is created using Boto3, which allows the application to access AWS Bedrock models. The embedding model is initialized using the Titan embedding model, with the model ID configured according to AWS Bedrock’s specifications.

For data ingestion, a function is created to load PDFs from the data folder using PyPDFDirectoryLoader, split the documents using RecursiveCharacterTextSplitter, and return the resulting chunks. Another function handles vector store creation using FAISS, taking the chunks and generating embeddings with the Bedrock embedding model, and saving the resulting index locally for future retrieval.

Next, functions are created to load different LM models like Cloudy or Llama 2. Each function specifies the model ID, Bedrock client, and model arguments (like max tokens, temperature, and max generation length). These functions return the LLM object for further use. The prompt template is defined using LangChain’s PromptTemplate, specifying that the model should provide concise, detailed answers of at least 250 words and should not fabricate answers if unknown.

The response function (get_response_LM) takes three parameters: the LLM model, the vector store, and the user query. It uses RetrievalQA from LangChain to perform similarity search on the vector store, retrieve relevant chunks, and feed them into the LLM along with the prompt template. The result is then returned as the output.

Finally, the application is wrapped in a Streamlit interface. The sidebar contains a button for vector store update, which triggers data ingestion and vector embedding creation. Another button allows selecting the Cloudy output, which loads the vector store locally, retrieves the Cloudy LLM, and fetches answers for the user’s query. Similarly, a button can be created for Llama 2 output, following the same process. The first time vector creation occurs, files are saved in a folder called files_index, containing index.files and index.pickle. Subsequent queries use this local index for faster response.

This complete project demonstrates an end-to-end workflow: ingest PDFs, create vector embeddings, store them in a vector store, integrate AWS Bedrock models via LangChain, and build a user-friendly Q&A application using Streamlit. The scalability of AWS Bedrock allows the use of multiple models, making the system adaptable for various real-world applications. By following this step-by-step approach, one can develop a robust, production-ready document Q&A application from scratch, fully leveraging cloud-based LLMs and embeddings.

**D) End To End Blog Generation Gen AI Using AWS Lambda And Bedrock**

We are continuing our generative AI on AWS Cloud series. In this specific session, we are going to develop an amazing end-to-end generative AI application using various AWS services. I will give you a brief architecture of what we are going to implement, and then step by step, I will show you how you can use the AWS ecosystem to implement an amazing generative AI application.

For this use case, we are going to keep it as blog generation. Our main aim here will be to create an API. We will use Postman to hit this API with a specific body that includes the user’s query, such as what blocks or topics we want to generate. In AWS, we use Amazon API Gateway to create APIs, so this will be the service we use. Once this API is triggered, it will hit our Lambda function.

This Lambda function will interact with Amazon Bedrock. If you don’t know about Amazon Bedrock, it provides access to multiple foundation models such as LLaMA, Amazon Titan, Claude by Anthropic, Command by Cohere, Mistral, and even Stable Diffusion for images. The Lambda function will take the user query, send it to the foundation model, and get back a response. In this case, the response will be our generated blog of 200–300 words. Finally, the Lambda function will save this content in Amazon S3 as a text file or PDF with a timestamp.

To get started, you need an AWS account. Go to Amazon Bedrock and click Get Started. You will see the available foundation models. You can do chat, text, or image generation here. For example, if you want to use LLaMA 2 Chat, you can select it. Each model may require access approval depending on the region. You can manage model access by selecting the models you want and clicking Save Changes. Once access is granted, you can invoke these models.

Next, let’s look at AWS Lambda. Lambda is a serverless compute service that runs code in response to events without managing servers. Instead of using an EC2 instance, which requires setup and maintenance, Lambda allows you to deploy your code directly and scale automatically. Lambda is cost-effective because you pay only for the number of requests and execution time.

You can create a Lambda function from the console. Give it a name, for example, AWS_App_Bedrock, and select Python 3.12 as the runtime. After creating the function, you can add triggers like API Gateway. Inside the Lambda function, you will write the code to invoke Amazon Bedrock models. Although the code editor in AWS Lambda is available, it is better to write the code in VS Code and copy it over for better code suggestions and readability.

In VS Code, first, create a virtual environment with Python 3.12. Install required libraries like boto3 using a requirements.txt file. Boto3 will be used to invoke the foundation models. After setting up your environment, create an app.py file and start coding. Define a function like blog_generate_using_bedrock that takes a blog topic as input and returns the generated blog as a string.

Next, write your prompt for the model. For LLaMA models, use the inst keyword and wrap the user query between Human: and Assistant:. For example, “Write a 200-word blog on the topic <blog_topic>.” Then construct a request body with parameters like prompt, max_gen_length, temperature, and top_p.

Use Boto3 to create a Bedrock runtime client with proper configuration, such as read timeout and max retries. Call invoke_model with the model ID and JSON-encoded body. Read the response using response.get('body').read(), and convert it to JSON. The generated blog content will be available in the generation key of the response. You can print it or save it to S3.

This Lambda function will be triggered whenever the API Gateway endpoint is called. The function will send the user query to Bedrock, receive the generated content, and store it safely in S3. This setup provides a fully serverless, scalable architecture to generate content using AWS’s foundation models.

And finally, I’m going to return the blog_details.

So this function here, you can see it handles the blog generation. I’m also going to write my try-except block. I’ll create an exception as E and just print the errors. Understand one thing: whatever we print in AWS Lambda gets logged in CloudWatch, which is a great advantage because you can always check the logs there. So if there’s an error generating the blog, it will print that error, and I’ll return a blank response.

Here, we have basically done almost everything. This is the blog detail that we are getting, and this function is responsible for generating the blog. Now, I’ll define the lambda_handler. Whenever the API Gateway sends a POST event, it will hit this function first. We have to capture the event along with the context and retrieve whatever query we are sending. In this case, it’s the blog topic.

Inside the handler, I’ll parse the event using json.loads(event['body']). Then I’ll create a variable blog_topic to store the topic from the request body. For example, blog_topic = "machine learning". Next, I’ll call the function generate_blog_using_bedrock(blog_topic) which interacts with the LLaMA 2 model to generate the blog. Once the blog is generated, I’ll save it in S3 as a .txt file using the current timestamp to make each file unique.

I’ll import datetime to create the timestamp and construct the S3 key as blog_output/{current_time}.txt. I’ll specify the bucket name, for example aws_bedrock_course_one. Then I’ll create a function save_blog_details_s3(s3_key, s3_bucket, blog_content) which uses boto3 to put the object into S3. The body of the object will be the generated blog.

Once everything is done, I’ll return a status code 200 and a message "blog generation is completed" for verification. This completes the Lambda function logic.

One important thing is that Lambda uses an older version of boto3 by default, which does not support Amazon Bedrock yet. So we need to create a Lambda layer with an updated boto3 version. To do this, create a folder named python, install boto3 using pip install boto3 -t python/, zip the folder as boto3_layer.zip, and upload it as a custom Lambda layer. Once the layer is created, attach it to your Lambda function to update the packages.

After the Lambda function is ready, create an API Gateway with an HTTP API. Add a POST route, for example /blog_generation, and integrate it with the Lambda function. Deploy the API to a stage, like dev. Then you can call this API via Postman. Make sure the JSON body contains the key blog_topic with the topic value.

Finally, create the S3 bucket if it doesn’t exist yet. For example, aws_bedrock_course_one. When the Lambda is triggered via API Gateway, the generated blog will be saved in this bucket as a text file. You can check logs in CloudWatch to debug any errors, like permissions issues. Make sure the Lambda execution role has permission to invoke Bedrock models and access S3.

Once everything is configured correctly, hitting the API generates a 200-word blog using LLaMA 2 and saves it in S3. You can download the text file to see the blog content. This setup gives a full end-to-end generative AI workflow using AWS services, ready to scale or integrate with more advanced features like vector databases in future projects.

**E) Deployment Of Huggingface OpenSource LLM Models In AWS Sagemakers With Endpoints**

In this video, I'm going to show you how you can actually deploy your Hugging Face model, whether it’s an open-source model or any other language model. Specifically, we will be working with AWS SageMaker. Amazon SageMaker is one of the key services in AWS that allows you to complete the entire lifecycle of a data science or AI project, including MLOps and deployment.

In this video, I will demonstrate how to take any Hugging Face model and deploy it directly in SageMaker. Make sure to watch till the end. One important thing to keep in mind is that some charges may incur while creating endpoints. So, please be cautious, and once your endpoints are no longer needed, make sure to delete them. I’ll provide full guidance as we go along.

First, we start by searching for AWS SageMaker. I will also cover how to work with SageMaker Studio. You can click on “Getting Started.” The AWS SageMaker documentation is excellent. For those new to it, SageMaker provides machine learning capabilities for data scientists and developers to prepare, build, train, and deploy high-quality ML models efficiently.

We will go step by step. First, I go to the domain section to create a domain. There are two options: set up for an organization or for a single user. For this demo, we will choose single-user setup. SageMaker automatically creates an IAM role with full access policy, public internet access, standard encryption, access to SageMaker Studio, SageMaker Canvas, and IAM authentication.

Once you click to create the domain, it takes some time depending on usage. After creation, you’ll see the domain listed as pending and, eventually, it will be ready. You can also add multiple users if required. The launch button allows access to Canvas, TensorBoard, and the full Studio ecosystem.

In Studio, you can access Jupyter Lab. This environment allows you to quickly start deploying, fine-tuning, and evaluating pre-trained models, especially for language models. You can also perform AutoML and model evaluation. For this demo, I will create a Jupyter Lab space named “Test Demo SageMaker” to work with Hugging Face models.

Next, you select the instance type. Different instances have different pricing. For generative AI models, large GPU-based instances are often needed, but for this demo, we will use a smaller system. I selected ml.m5.2xlarge with 8 CPU cores and 32 GB memory. Storage is set to 10 GB. Once you click “Run Space,” the environment will be ready for coding.

Inside the Jupyter Lab, the first step is to install the latest SageMaker SDK using pip install sagemaker. This updates your environment to the latest version. We then import SageMaker and Boto3 and set up the session with sagemaker.session.Session(). SageMaker automatically creates a default S3 bucket for data, models, and logs if it doesn’t already exist.

Role management is crucial. Since SageMaker executes code using AWS services, the IAM role must be properly configured. Using Boto3, you can fetch the execution role and handle exceptions if the IAM user is not set. Once the role is set up, you can create the SageMaker session and print the session region and role ARN to verify everything is working.

After setting up the environment, you can load any Hugging Face model. For this demo, I used DistilBERT uncased distilled squared for question answering. We define the Hugging Face Hub configuration and create the model using sagemaker.huggingface.HuggingFaceModel(). The model parameters include the role, transformer version, PyTorch version, and Python version.

Deployment is done using the deploy() method of the Hugging Face model. You specify the instance type and count. Input data for inference must follow the model’s expected format, including question and context. Using predictor.predict(data), you get the model output. For example, asking “What does Chris like?” returned “data science.”

You can view deployed endpoints in SageMaker Studio under Deployments. Each endpoint can be tested using a JSON body with the input data. Auto-scaling is also supported depending on the workload. This process allows you to use the deployed endpoint anywhere in your code.

For larger models, you can call the Hugging Face deep learning container using the image URI. This is useful when running models in Docker containers. For example, Falcon B models with multiple GPUs require high-end instances with significant memory, CPU, and GPU resources, which incur higher costs. That’s why we used a smaller model for this demo.

The steps, in short, are: update SageMaker, configure the role and region, call the SageMaker image URI, initialize the Hugging Face model, deploy it, and then use structured payloads for inference. The documentation and GitHub examples provide additional guidance.

Overall, this demo shows how to deploy a Hugging Face model in SageMaker safely, keeping costs in mind, while providing a full development ecosystem. You can extend this knowledge to larger models and production-level deployment once familiar with the process.

I hope you found this video helpful. I will provide all code examples and relevant links in the description. Refer to Hugging Face and SageMaker documentation for additional labs and use cases.

# **XXII) Getting Started With Nvidia NIM and Langchain**

**A) Building RAG Document Q&A With Nvidia NIM And Langchain**

In this video, I’m going to show you some of the amazing and powerful features of Nvidia NeMo, which was recently announced by Nvidia.

So, what exactly is Nvidia NeMo? It’s a breakthrough in generative AI development. Essentially, NeMo is a set of inference microservices for deploying AI models. It completely revolutionizes how enterprises can deploy generative AI.

Along with this, Nvidia NeMo offers multiple AI models — it can be a single model or multi-model. Not only that, it provides you with Nvidia AI foundation models, which you can integrate into your applications just through APIs. It’s highly scalable and seamless to use.

In this video, I’ll not just explain this, but also show multiple coding examples, so make sure you watch till the end.

The clear point is — many LLM models will keep coming, but the winner will be the company that provides the best inferencing solution.

Getting Started with Nvidia NeMo

Here is the Nvidia NeMo page. You can instantly run and deploy generative AI models, explore community-built AI models, or access models optimized and accelerated by Nvidia. You can deploy anywhere with Nvidia NeMo.

It’s easy to integrate — just a single API call and you’re ready to go. Also, if you want to try it out, Nvidia gives you 1,000 free credits upon account creation. That’s more than enough to explore and call multiple models.

Exploring the Models

Once you click “Try it Now”, you’ll see all the available models:

LLaMA 3 70B

Nvidia Foundation models

Open-source models

They cover multiple use cases: Reasoning, Visual design, Retrieval, Speech, Biology, Gaming, and more

API Key Setup

Before starting your project, you need an API key. For example, if you want to use LLaMA 3 70B instruct, click on the model, and you’ll see a chat interface.

Click “Get API Key” (green button)

This key will authenticate your Nvidia AI foundation endpoint for testing and evaluation

Make sure to store it securely; you’ll need it for coding.

Setting Up the Environment

Open VS Code and create a conda environment:

conda create -p venv python=3.10
conda activate venv


Create a requirements.txt file with packages you need, for example:

openai
python-dotenv


Install packages:

pip install -r requirements.txt

Using the OpenAI Client with Nvidia NeMo

Import and create a client:

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


Create a chat completion:

completion = client.chat.completions.create(
    model="llama-3-70b-instruct",
    messages=[{"role": "user", "content": "Provide me an article on machine learning"}],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    print(chunk, end="")


Notice how fast the inferencing is — this is where Nvidia NeMo shines.

Building an End-to-End RAG Application with LangChain

Next, let’s create a RAG (Retrieval-Augmented Generation) application using LangChain and Nvidia NeMo.

Update requirements.txt to include:

langchain
nvidia-nemo-endpoints
langchain-community
streamlit
pypdf


Install the packages:

pip install -r requirements.txt

Setting Up the Project

Load PDFs from a folder (e.g., US Census PDFs)

Create embeddings using Nvidia embeddings

Split documents into chunks for better retrieval

Example code snippet:

import streamlit as st
from nvidia import embeddings, chat_nvidia
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

Load API key
api_key = os.getenv("NVIDIA_API_KEY")

Initialize model
llm = chat_nvidia(model="meta-llama-3-70b")

Load PDFs
loader = PyPDFDirectoryLoader("US_Census")
documents = loader.load()

Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

Create embeddings
embeddings_client = embeddings.NvidiaEmbeddings(api_key=api_key)
vector_store = embeddings_client.create_vector_store(split_docs)

Streamlit Interface

Add buttons to trigger document embedding

Ask questions using retrieval chains

Display answers with context

st.title("Nvidia NeMo Demo")

if st.button("Document Embedding"):
    # Create vector DB
    st.session_state['vector_store'] = vector_store
    st.success("Vector store DB ready with Nvidia embeddings!")

query = st.text_input("Enter your question from documents:")
if query:
    qa_chain = RetrievalQA(llm=llm, retriever=st.session_state['vector_store'].as_retriever())
    answer = qa_chain.run(query)
    st.write(answer)


Example question:

“What is the difference in the uninsured rate by state in 2022?”

The app fetches the answer from the documents using vector similarity search.

Conclusion

Nvidia NeMo allows fast inferencing and seamless integration with LangChain

Supports multiple AI models and foundation models

Enables end-to-end RAG applications easily

Perfect for enterprises looking to scale generative AI deployments

Definitely explore Nvidia NeMo and all the available models for your specific use cases: reasoning, visual design, retrieval, speech, and more.

That’s it for this video!
I’ll see you in the next one. Have a great day ahead!

# **XXIII) Creating Multi AI Agents Using CrewAI For Real World Usecases**

**A) Youtube Videos To Blog Page Using CrewAI Agents**

Welcome to this crash course of creating multiple AI agents for real world use cases using Q AI if you haven't heard about this platform.

So Q AI is an agent framework, uh, which will actually help you to create multiple agents for various amazing use cases.

Um, so in this video, I'll be talking about this entire platform. Along with that, I will also show you with an amazing use cases. Uh, I will try to create a complete end to end use case and how I can use this Q AI platform or framework to develop multiple AI agents and where it will basically be required.

See in long chain. Also you have an option to create agents, right? But here the main thing with respect to AI is that here your agent will be able to communicate with each other, right, in an efficient way so that we will be able to make or will be able to implement the task much more in an efficient way.

Okay, so this is the entire web page of the crew. I. And right now many, many people are using it. They are creating multiple agent crews. You can see over here in last seven days all these things are there. You'll also be able to work with the open source tools that is already provided over here. And in this video I'll be also talking about that and showing you with the most practical use case.

Um, so let me just go ahead and let me just start. First of all, we'll understand what kind of use case I will be solving. Okay.

So let's say I have a project over here and everybody knows that I have a YouTube channel. Now in my YouTube channel, there are more than 1900 plus videos. Um, now, what I really want is that, uh, for every video, I need to have a blog page. Let's say I want to probably create a blog platform right now. I want to create a blog platform. Just let's imagine. So what I will do, I will take up every video of mine, probably based on the content. I will go ahead and write my entire blog page itself, write whatever content is required in that blog page.

Now, this task is a really tedious task. Right now, what I'm actually going to do is that I'm going to automate this entire thing with the help of AI framework, where this blog platform will be automatically created from all my YouTube videos. With respect to all the things that I have said in my YouTube videos, that is most important.

Okay, so with the help of Q AI, what we are really going to do is that first of all, let's say that if a user queries any kind of, uh, videos with respect to, I let's say I'm querying about Q AI over here. So what it should happen is that it should go to my YouTube channel, pick that particular video, extract that entire content and based on that particular content, summarize that information, put it into blog page.

Right. Now, obviously, I know that I have created more than 1900 plus videos now if I really want to probably work with multiple people over here. So I need to have a content writer. I need to have a researcher, right, who will probably go and explore every video in my YouTube channel based on the query it is writing, and then it will go and validate that content. And after probably validating the content, I need to have a content writer separately who will specifically write this particular blog page. Also right now here, um, just to understand, for just one video it is fine. But if I have 1900 plus videos, it is going to take a lot of time because validating is also there. I need to probably go and check the content proofing, do multiple things over there, right?

So with the crew AI, we can automate this thing completely in a whisker of time. Uh, and that that is where we will understand how this AI agents. Okay. And here also you can see, right, if I'm involving multiple people, they also need to communicate with each other. Right. Then again there may be some kind of communication gap. You know how they are specifically going to work. So with respect to that you are going to take a lot of time over here right now with the help of Creo AI. So Creo AI, if I talk about they are three main important components. One is the agents, one is the task and one is the tools.

Okay. Now here if I probably consider this specific use cases as I'm saying. First of all, we need to explore my YouTube channel, right? So I need to have a researcher over here who will probably go and query any query in my YouTube video, and it will probably bring that particular YouTube video. Okay. After seeing that particular YouTube video, he needs to listen. He needs to probably hear out what you have about that entire YouTube video itself. So here also we require an expert right, expert who is good at data science, who is good at data analysts because my entire channel is based on data science itself, right? I so here obviously I need an expert person who will probably go ahead and watch each and every video and retrieve the content out of it. Right.

Then after that, I need to pass it to my content writer. Right. So content writer. So here I have a domain expert person, right? It can be a data scientist. It can be an analyst. Right. So these two roles here you can basically consider in Q AI these are nothing but these are agents. Okay. So we are specifically going to take consider this as agents. So agents are none other than the people who are having some kind of experience with respect to the work. Right. It can be a domain expertise like in data science domain. It can be data scientist, data analyst. It can be a content writer. Right. So it can be anyone. Right. So this basically becomes an agent.

So in the Q AI the first component is specifically agents. The second component that we specifically use in Q is task. Right now each and every agent does some kind of task. Let's say the domain expert people will probably go ahead and watch this particular YouTube video, try to translate, uh, take out the entire information that I've told in the YouTube video itself. Right. So, uh, that basically becomes the task for this particular agent, right? Similarly, content writer is the agent. Its task is to write the content. Right. So that can basically be a task. So this is the most important second component. Like what task a agent is specifically doing that also we need to define.

And the third thing over here is about tools okay. Let's say, uh the domain expertise is over here. I do not have a domain expertise. Let's consider. So if that domain expert is one some kind of help, right. Uh, like, uh, let's say once it is searching from the YouTube, uh, video itself. Right. Uh, any of my video, it is probably searching, let's say that it wants to get the transcript of the video. So what may happen is that this person may use some kind of tool to do that. It can be a third party tool. It can be an API. It can be anything as such. In this particular case, uh, let's say I want to get the transcript of the my YouTube video. Then I may use a transcriber. Right. And that transcriber can be a third party tool itself. Right. And that is where a tool comes into existence. Like with what will be the main, uh, what will be the main way of probably let's say that if I have some kind of dependency on some third party tools itself to explore this particular task, right. How can I perform this particular task with the help of this particular tool? So I may also have a separate tool which will be able to provide me the transcription of the entire video. Right. So that is where we can specifically use the tool. Other tool can be okay, I want to probably do a Google search API right. Google search. So this can be another tool. Right. So similarly multiple tools will basically be there which we can actually use. So agent will have some specific task. And this task can be performed by a specific tool. Okay. So this is how it actually works.

Now similarly here you can also see I have created two agents. One is researcher, one is content writer because that is what I actually require. Researcher will explore the videos from this particular videos. It may use some kind of IT tools to transcribe the entire content. Right. Analyze the entire content. Once that thing is done, then that particular researcher will pass it to the content writer because it now has the entire information. This entire info will be basically there after performing this particular task. Then once it provides it to the content writer, the content writer, what will happen based on the research? It will write the blog page, right? And finally, this will basically be my output, right? So here you can also see that interaction is there between the researcher and the content writer. Right. And this this process. Now see, this is one way. So this process is entirely called a sequential process. There are also other processes which is called as hierarchical processes. So here in the sequential process once the researcher completes his work, it is going to give that entire output to the content writer and the content writer further based on the task it is assigned, it will probably go ahead and create the blog page.

So I hope you got an idea about what are the main important components of Creo AI. One is the agent, one is the task and one is the tools. And based on this we can automate this entire use cases. Now let's go ahead and implement it practically. As I said in the agents I'm going to probably create two agents. One is researcher, one is the content writer. Then I'm going to go ahead and define the task for this specific agent. Like explore videos. It can be explore videos. It can be exploring. Another thing it can be probably exploring Google Search API, anything it can be. Now to complete this particular task, we have dependency on some tool which is called as YT tools because, uh, at the end of the day, I require the transcription or of my entire YouTube video. So for that I may require a tool. And this can also be a custom tool, which you can also create by yourself. Right? So once we get this particular tool, we will be able to do this particular task. And after that we will be able to complete the researcher work. Once this researcher work is completed, we will pass this entire work to the content writer, because the content writer needs to write the entire blog page based on the research, right? So this entire process is also called as sequential process because once this is getting completed, the next step is to get to complete this. There are also other processing, other processes like hierarchical process where parallelly also you can actually do this particular task.

Okay. Now let's go ahead and implement this entire project completely from scratch. So guys, I have opened my new project over here in my VS code. I will go to the terminal. The first step is probably to create our conda environment. Okay. And this you really need to do it for every project. So I will go ahead and write <code>conda create -p venv python=3.10</code>. So okay I will be taking 3.10. That is 3.10. Now after doing this installation probably happen once the installation actually happens on a new environment is basically created. So what we are going to basically do is that we are going to create our <code>requirements.txt</code> file. So let me just go ahead and write my <code>requirements.txt</code> file. Okay. Uh now inside this <code>requirements.txt</code> file I will be using some of libraries that I really need to install. One is the CRI. Then one I'm going to write Python. Okay. Right now I don't require this. So let me just go ahead and write <code>creo_ai_tools</code> right. So these are the two important libraries I will be requiring. So let me save it. Now this environment is basically getting created or it has got created. So I will go ahead and activate <code>conda activate venv</code>. Now let me quickly go ahead and let me do one thing guys. Let me hide my face so that you will be able to clearly see this okay. Now let me just quickly go ahead and write <code>pip install -r requirements.txt</code>. And this all both the requirements will get installed okay.

Okay, so once this installation is basically happening, uh, we will continue our task. First of all, as you know we need to create our agents. So <code>agents.py</code> I will go ahead and create it. The next will basically be <code>tools.py</code>. Okay. Uh agent tools and I also require task okay. <code>task.py</code>. So these three components are we really need to create. So first of all I will go ahead and create my agents. Now for creating the agents. First of all, I will go ahead and import from Creo I import agent. Okay. Now, uh, since, uh, you know that, uh, we really need to create some kind of agent over here, right? And for creating an agent. So first of all, what all things I will create. So I will probably create a, uh, senior blog content researcher. Okay. So this will basically be my first researcher, which will be an agent who will be doing my task. Okay. So these are like people who will be handling all my tasks. Okay. Now let me quickly go ahead and create my researcher. So this researcher will be my blog researcher. Okay. And this blog researcher will be of type agent. Okay. Now with respect to agent, uh, there are some important parameters that we really need to use. Okay. First parameter we really need to give is role. What kind of role it is basically doing or what kind of role it needs to do it okay. So here I'm going to basically say that okay, these are nothing, but they have to probably be a blog creator or blog researcher from YouTube videos okay. YouTube videos. So these are some default information that we really need to give. One is the role. The other one is goal. Okay. So here we also need to specify the goal over here. Uh here we can basically write get the relevant, relevant video, uh, get the relevant video, content for the topic, whatever topic I say. Okay. From my channel. Okay. So this will basically be my role for this particular agent.

You need to get the relevant video, content for the topic. This from my channel. Okay. Name. I can basically write over here description so I can. Right. So let me just go ahead and write. So let it be. I don't require this too, but it is getting suggested by the, uh, Amazon whisper that I have specifically used over here, but it's okay. I don't want this. Then I will go ahead and set up my verbose. So verbose will be true, which will be able to see some information out over here. We're also going to set one parameter which is called as memory true. So which will be initialized with some memory. And we will go. Also go ahead and write some backstory about this particular agent okay. So let's go ahead and define some backstory. So uh I will keep a backstory something like this. This person or this agent is an expert. Okay, see? See this? Okay, so I will just go ahead and write. This person is expert in understanding videos in AI, data science, machine learning and gen. I am providing suggestions. Okay. So this will basically be my back story. And then the third thing that I actually require is my tool, whether I'm going to use some tools or not. So here I'm going to basically define my tool. And right now I'll keep the tool empty because I have not created any tool. Okay. And then further I will also say allow delegation. Delegation basically means will I be transferring after whatever work that I do or this agent does to someone else. So we will set this allow delegation to true. Okay. So this is all the default parameters that we really need to write for an agent. Okay. Uh, and based on this you can create anything.

# **XXIV) Hybrid Search RAG With Vector Database And Langchain**

**A) Introduction To Hybrid Search**

Hello guys. So in this video we are going to discuss about something called as hybrid search. Okay. And this kind of search specifically we use in our Rag applications also. Right. So till now if you are following all the modules, the rag that we have specifically created, you know, let's say that if I have documents, let's say if this is my documents, then what we do is that we try to divide these documents into chunk of documents. Let's say this is d1, this is d2, this is d3. Like this we usually divide into multiple chunks of documents. Right. And this is dn.

Now after dividing all these documents into chunks of documents then we specifically convert this into vectors. Right? So here we are basically converting into embedding vectors. Then after we convert all this text to vectors, the next step is that we store all these vectors in some kind of vector database. Right. So let's say one of the vector databases that we have already seen, vector database. We also say it as something called as vector store DB. Right. Now once this entire information is basically stored over here, it will be stored in the form of vectors. Right. We will be having some kind of numbers over here itself. Right.

Now, when a user queries, whenever a user queries right over here, then what happens? Usually in this particular scenario this particular query is also converted into vectors. And then a vector dense search or vector search is basically done. Vector search is basically done in this particular vector database. So once we have this particular vectors we do a vector search from this particular vector database. And here are some of the algorithms like over here one of the algorithms that may take place over here for basically to match the exact vector search, we can actually use cosine similarity. Right. And based on this cosine similarity check, we will be able to get the response.

And once we get this response we can combine this with our prompt template plus LM model and finally we can summarize any information as my final output. Right. So in this scenario you will be able to see that the kind of search that we are doing right, the vector dense search. This search is basically called as semantic search. Semantic search basically means we are trying to find out the similar vectors from this particular vector database, and based on that, when that similarity search is done, we are able to get the response. And this kind of Rag, most of the Rag application uses this kind of search.

And if I talk about vector databases like Face, Chroma, right, they have this entire feature like this. Right. But I really want to talk about one more amazing search mechanism, which is called as hybrid search. Okay, so here I have this hybrid search. Now inside this hybrid search here we are just not focusing on only vector dense search mechanism. Right. Here we specifically combine multiple search techniques. Right. And what kind of techniques we will specifically apply over here? We will be combining multiple search techniques.

Now specifically if I really want to divide all this particular technique, one is the semantic search, the second one is something called as syntactic search. Okay. Now I will talk more about it, what exactly are these kinds of search basically. So one is semantic search, one is syntactic search. Semantic search specifically we say it as a dense vector search. So here we are specifically trying to search elements that are similar right. And obviously when we are using this dense vector search we basically are checking out the similar kind of content.

If I probably say with respect to syntactic search, right, or I can also say this as exact search, I can also say this as keyword search. Right. And if I also want to probably provide some more names like exact search and keyword search, and this usually happens with respect to a sparse. Or I can also say this as sparse vector search. So here you could see in semantic search I have a vector store right, which has all the vectors embedded over here. Right. So here all my text is basically converted into dense vectors. And this dense vector is basically stored in my vector store DB.

In the case of syntactic search, you will be able to see that we will be having sparse matrix. So all this text will basically be converted into sparse matrix. Now if you remember or if you know right what is sparse matrix and how do we get a sparse matrix, if you remember if you know about these techniques like one hot encoding, bag of words or TF-IDF, right, whenever we try to use these techniques, we will be able to convert our text into sparse matrix. Now, when we say sparse matrix, that basically means there will be maximum numbers of zeros and ones, right, wherever a specific word will be available in a sentence that will be one and remaining all will be zeros. Right. So this is basically called as a sparse matrix.

And with the help of this sparse matrix here we are just focusing on a specific keyword like this keyword. Right. Whatever keyword we are actually trying to search. And this sparse matrix also has some very important properties because from this we will be able to probably do the keyword search itself. And this is nothing but it can also be considered as an exact search. So in the case of hybrid search, it is just going to combine both these particular techniques, that is semantic search and syntactic search. Or I can also say this as exact search or keyword search. Right. It is just not going to be dependent on one of the search mechanisms.

And similarly there are different different kinds of databases that are available just to show you as an example, like there is a database like Cassandra. Right. We will be using a library called as "cashier". Right. And with the help of this cashier you will be able to see that we will be able to access the Cassandra database. I will show this as we go ahead. Okay. But right now, just try to understand over here what exactly Hybrid Search is doing. Right? It is combining multiple search methodologies. And if I probably talk about in most of the e-commerce websites and all, hybrid search is definitely implemented in the backend.

Now let's go ahead and see how does a hybrid search work. Okay. So here we will go ahead and write how does hybrid search work? So let's say in the case of hybrid search, what happens? Let's say if I have specific documents, for all these documents when we are storing this, when we are storing this in our vector database, let's say this is my vector database, okay? And this vector database supports both. Right. What does it basically support? It supports vector search, dense vector search. Along with this it also supports keyword search. It has both this particular property.

Now whenever I have this particular document inside this particular database or inside this particular vector store DB, this document will also be converted into a sparse matrix. Right. By using techniques like TF-IDF or bag of words. So this kind of embeddings it will be converted and it will be stored inside this particular vector database. Along with that, for the same documents, we will also try to convert this into vectors, which will be my dense vectors. Dense vectors do not have maximum number of zeros or ones. Okay, that is the difference. And we will try to store these particular vectors inside this particular vector store DB.

And again here you can use different different techniques, or the techniques can be anything as you want, like we have seen. Right. We have used different different embedding techniques. We have used OpenAI embedding techniques, we have used Llama embedding techniques, or if you want Hugging Face embedding techniques. So by all these embedding techniques we can convert our text into dense vectors and we can store this in this particular vector database. Now the idea is very simple. Since we are storing this in both these particular formats, sparse matrix and dense vector matrix, whenever a user puts up any kind of queries, now with respect to this particular query, I will be able to retrieve two different kinds of results.

One result will be something related to my exact match or exact search or keyword search. And with respect to this keyword search, let's say whatever results I am getting, it will be my top K results. Right. This will be based on this particular keyword search or exact search. So here sorry keyword search. I have written it downwards. Right. So here you will be able to see that whenever a user probably queries any information over here from this particular vector database, first of all, for the user query again their query will also be divided into two types. It will be converted into two types, one is the sparse vectors and the other one is something like dense vectors. Right.

So both these vectors it will be converted. Then this part based on the sparse vectors, here we are specifically going to get the keyword search. And here we are going to get the results. Right. Let's say the top K results I will be getting over here. Based on these dense vectors, I am actually going to do the vector search. And again I will be getting another result over here. Right. Let's say this is my result one, top K result one. And this is my top K result two. Okay. Now top K result one and top K result two are specifically there. This is basically done on the vector search which is nothing but it is a semantic search. This is basically a semantic search, similar kind of elements. And this is nothing but this is a keyword search. Right. Based on the sparse vectors that we have actually taken.

Now in hybrid search, what we do is that we combine both of them based on weightage. Okay. And then we finally get the final response. Now the question arises how we are going to probably combine this particular result one and result two to weightage based on this information that we have got. And finally, when we get the response, we can then further combine this with our LM model and summarize this entire data. Okay, that is something separate. But still the question arises how do we combine result one and result two, and how based on weightage we take our final result. Right. Final result.

And that is what we will be seeing in our next video. The technique is very simple and the technique name is something called as reciprocal rank fusion in hybrid search. Okay. In hybrid search. So this technique we will be discussing in our next video which will be the core behind this particular hybrid search. But the idea is very simple. In hybrid search we take a combination of two different things. We try to do the search based on semantic syntactic, which will be the combination of dense vector search and exact search or keyword search. Okay. So both these particular techniques we basically use.

So yes, this was it. In the next video we are going to discuss about the reciprocal rank fusion in hybrid search and how we can basically combine both these particular results based on some kind of weightage. Right. When I say top K result one, top K result two, that I may get five results. Right. Based on the keyword search, I may get five results based on the vector search, something like that. Right. So yes, this was it. I will see you in the next video. Thank you.

**B) Reciprocal Rank Fusion In Hybrid Search**

So guys, now let's go ahead and proceed with the discussion with respect to reciprocal rank fusion and hybrid search. As I said, whenever we try to query or whenever a user queries from this particular vector database, which supports both vector search and keyword search, which is a combination of semantic search and exact search over here. First of all, to get the keyword search, we have to convert this into sparse vectors, and the sparse vectors will be probably searching from this particular vector database, and we get top k results.

Similarly, with the help of dense vectors, we also get a vector search. And here we are going to get a result that is top k results. Okay. Now combining both of them we have to probably consider a response. Right. How to probably get the final response. That is what I'm actually going to discuss now.

Okay. So here, with the help of reciprocal rank fusion and hybrid search, we will be considering two important things. Okay, let's say this is my vector database. Okay. And from this vector database, you know that whenever a user queries. So based on this query, let's say if this is my dense vectors, first we are going to do the vector search, and we are going to get the top k results, right. So when I say top k results, let's say that this is my first document. Right. And this can be my second document. Right. And this can be my third document. Like this I will be getting a lot of documents.

Similarly, over here, based on my sparse matrix search, that is the keyword search, I'm actually going to get my top k documents. So here also I will go ahead and draw this. So let's say here I'm going to get some other sentences. Okay. And the ordering will be different. Right. So here, let's say this is the order that I've actually got. But I got document one, two, three, four, five, let's say the five documents I've actually got. And here also we have got some other documents, so let's say some other documents. But here the numbering is different. Let's say the numbering is like 43251. Okay. The numbering is different.

This is my result one top k documents and this is my result two top k documents. Okay. Now considering this, see here I've got in this ordering, that is 12345 sentence. That is that I got in my top k like document one, document two, document three, document four, document five. Similarly, over here, document four, document three, document two, document five, document one. This is the ranks that we have specifically got. Now how to calculate the final score. Because here this is based on my semantic search. This is based on my keyword search. When I say semantic search, that is basically based on cosine similarity, okay. So we are trying to find out the similar vectors and then we are trying to find out our final score.

So in order to calculate the final score, the formula is very simple. Here we'll write:

summation of 1 / (constant + rank of document)

Now this particular constant, that is C, it depends on various databases. Okay. Now with respect to various databases, this may range between 1 to 60. Okay. And this is a constant case. In various databases they'll set this value to 10, 20. And some of the databases, I've seen the value to 60. Okay. So since this is a constant, we are just not going to focus much on this. Instead, I will just take up this formula. Now summation of i is equal to 1 / rank of t. Okay. Since this is a constant, for calculating every rank of every document, it will probably give you the same answer with respect to that particular constant; same impact it may be creating.

Now how do we compute the rank, right? So for this particular sentence, you'll be seeing 1 / rank of D. So if I go ahead and compute this, this is nothing but 1/1, okay. This is nothing but 1/2 because this is rank two, right. This is rank three, so I'll get 1/3. It is nothing but 0.33. Then here I have 0.25. And this is my fifth document, let's say the rank is five. I'm actually going to get 0.2, okay. Similarly, in this particular case, this is my document four. Right. And if I try to compute this particular rank, it will be again 1/1, 1/2, 1/3, 1/4, 1/5.

So that basically means the document D4 is basically getting a rank of one in this keyword search here. D4 is basically getting 0.25. Okay. Now when we try to combine the final score, okay, this is important. See this. So if I finally go ahead and take up document one, so if this is my document one and if I try to calculate the overall rank, then what we have to do, we have to do the summation. So let's see document one is giving 1 over here and 1/5. So I will just go ahead and write 1 + 0.2. So this will be 1.2, right.

Similarly, if I go ahead and calculate for document two, right. So document two over here gives 0.5, right. So here I'm actually going to get 0.5 + document two over here is 1/3, that is nothing but 0.33, right. So here you can see I'm actually going to get 0.83. So this is my score that I'm actually going to get. And obviously this score is better than this score, so I will be getting document one based on my 50% probability of 50% focus on keyword search and 50% on semantic search. Right. If you change the weightage, then you'll be able to see that some values may also change.

Similarly, you have this document three. Now in the case of document three, you'll be able to see that here I have 0.33. So let's calculate this: 0.33 + document three is nothing but 0.5. Again here I'm going to get the same value, see 0.83. Now both these ranks are same, right. So if I want to retrieve the top k result, obviously this will be shown first. And then based on this two, right, where the focus is more—whether it is on keyword search or semantic search—let's say if I say 70% focus is there on the keyword search, then over here you can see 0.5 value is higher than 0.33. So it is just going to display document three in the second rank.

Right. So this way, the weightage can also be provided in keyword search and semantic search. Okay. And similarly we go ahead and calculate each and every document. So finally you can see that here we can combine result two and result one top k elements. And then we can use this reciprocal rank fusion in hybrid search by assigning ranks. And we will be able to calculate a score which will be finally given as a response. And here we can also assign weightage. And this can be combined with a prompt template. And we can summarize with our LM model. Right.

So this gives an idea about the entire hybrid search. Right now, in this section, in this next section, what we are going to do is that we're going to also see a practical example of how a hybrid search is basically done. But before that, if you know about one more search, right? That is based on graph knowledge search. Okay. Graph knowledge search. So in this graph knowledge search, I hope you may have heard about graph DB. One of the most popular open-source graph DBs is Neo4j.

Okay. Now if we are specifically using graph DB to store the vectors, let's say if this is my database and this database is nothing but my graph DB. In the case of graph DB, I will be having nodes which will be connected with relationships. Right. So this can be my node one, this can be my node two, and it will be connected with relationships. And whenever a user tries to query, there are three types of queries that happen: One is keyword search, which you can also call syntactic search or exact search. Then, along with this, it also supports semantic search. And finally, you will also be able to see that it supports graph knowledge search. Graph knowledge search.

So it supports all these techniques, which is quite amazing. Just imagine, hybrid search supports only these two types. But now, with the help of graph knowledge search, we will still be able to create a very good efficient Rag application, right? With the help of graph DB. And this we will be seeing about graph DB in the upcoming videos, in the upcoming sections. But let's go ahead and see some practical example for hybrid search.

**C) End To End Hybrid Search RAG With Pinecone db And Langchain**

So we are going to continue the discussion with respect to the hybrid search. Here, we are going to implement a complete, amazing project with the help of hybrid search using LangChain. For this, we will be using a vector database called Pinecone. Pinecone provides both keyword search and semantic search capabilities.

First of all, we will set up our account and then, with the help of code, we will try to write everything and create our vector database itself. So, go ahead and sign up—it is completely free. The first 2–3 instances you can create completely for free. I have already signed up, so I will log in. You just need to go to pinecone.io. Once inside, you will see that I have already created a Pinecone hybrid search instance. There is some data available, but as usual, we will not use that; instead, we will create our new index.

Since I want to do everything with the help of code, we will create the instance programmatically. First, you need your API key. Just click on API key, copy it, and make sure it is set as your default. You can also rotate the key if needed. I will copy this key and start coding.

Next, we need to install the required libraries. I have a requirements.txt with many libraries. For LangChain, we need several, and for Pinecone, we need pinecone-client, pinecone-text, and pinecone-notebooks. I have already installed these to save time, so everything is ready.

Now, I will set my API key in a variable called API_KEY. You can also create an environment variable if you want. Since we are using LangChain, we will import the PineconeHybridSearchRetriever from langchain_community.retrievers. This class is responsible for both semantic and keyword search. Keyword search here refers to a combination of sparse matrix and dense vector search. To perform hybrid search, we need this retriever class.

Before using the retriever, we first need to create our index in Pinecone. We will import OS and pinecone, and then define an index name. For example, hybrid_search_langchain_pinecone. Please use lowercase letters to avoid errors. Then we initialize the Pinecone client using our API key.

Next, we check if the index already exists. If not, we create it using PC.create_index(). We need to specify the dimension of the vectors, which I set to 384 because we will use the Hugging Face sentence transformer embedding model (all-MiniLM-L6-v2), which generates 384-dimensional embeddings. For the metric, I use dot product, which works for both sparse and dense embeddings. Finally, we specify the serverless spec and the cloud region (us-east1 for AWS). Once this runs, the index will be created in our Pinecone account.

Now that the index is ready, we can create our embeddings. I import HuggingFaceEmbeddings from LangChain and set model_name to all-MiniLM-L6-v2. The embeddings will be 384-dimensional. Make sure to set your Hugging Face token in your environment variables as HF_TOKEN. This is required for the embeddings to work.

For the sparse matrix, we use Pinecone’s BM25 encoder, which is based on TF-IDF. We initialize the encoder and fit it on a list of sentences. For example, "In 2023 I visited Paris", "In 2022 I visited New York", and "In 2021 I visited New Orleans". After fitting, we can dump the sparse matrix values to a JSON file for storage or later use.

Next, we create the Pinecone hybrid search retriever. This retriever combines both dense embeddings (from Hugging Face) and sparse embeddings (BM25). We pass the embeddings, the sparse encoder, and the Pinecone index. Once executed, the retriever is ready, and it supports both dense vector search and sparse keyword search.

Now we can add text to our index. We insert all our sentences into the Pinecone index. Once inserted, we can verify that the data is correctly stored. You will see all three sentences in 384-dimensional vectors.

Finally, we can perform hybrid search queries. For example, if we ask, "With what city did I visit last?", the retriever will use both dense and sparse embeddings to return the correct results: "In 2023 I visited Paris", "In 2022 I visited New York", and "In 2021 I visited New Orleans". Similarly, if we ask "Which city did I visit first?", it will return "In 2021 I visited New Orleans".

This demonstrates a complete working example of hybrid search using Pinecone and LangChain. It combines both dense embeddings and sparse keyword search, providing accurate and efficient retrieval.

I hope you found this example helpful. I will see you all in the next video. Thank you.

# **XXV) Introduction To Graph Databases And Cypher Query Language With Langchain**

**A) Introduction to Graph DB Wwith Langchain**
In this video, and in the upcoming series of videos, we are going to discuss graph databases and see how we can create amazing generative AI applications using them. Specifically, we will be using LangChain to implement these applications.

Let me first set up the context. In this series, there are many important concepts that we need to cover. First, we will learn about knowledge graphs, which form the fundamental concept behind graph databases. Understanding knowledge graphs will give us the foundation needed to work effectively with graph databases.

Once we cover knowledge graphs, we will explore different open-source graph databases available in the market. While there are both paid and open-source options, we will focus on Neo4j, one of the most popular open-source graph databases. We will look at how to work with Neo4j, including how to insert data, create nodes, set properties, define relationships, and manage entities.

To interact with Neo4j, we use Cypher queries, similar to how we use SQL queries with MySQL. One session will be dedicated to Cypher queries, where we will spend around 15–20 minutes learning how to write and execute them effectively. This hands-on session will help you become comfortable with graph database operations.

After covering the fundamentals and Cypher queries, we will move on to developing RAG applications (retrieval-augmented generation) and other applications using graph databases, with Neo4j as our backend. This will demonstrate practical use cases of combining AI with graph structures.

Additionally, LangChain has introduced an exciting new module called LangGraph. With LangGraph, you can create multi-AI agents that perform complex tasks. These agents also leverage the concept of knowledge graphs, making it essential to understand the topics we will cover in this series.

Overall, this series will provide a strong foundation for working with graph databases and AI agents. It is going to be both interesting and highly informative.

In the next video, we will start with a detailed explanation of what knowledge graphs are and then proceed to other related topics.

I hope you find this video helpful. I will see you in the next video. Thank you!

**B) What is Knowledge Graph**

In this video, we are going to discuss Knowledge Graphs.

So what exactly is a knowledge graph? You can consider a knowledge graph as a semantic network. Please pay attention to these keywords, as they are very important. A knowledge graph represents a network of real-world entities.

For example, real-world entities could be events, situations, concepts, or even people and places. These entities illustrate the relationships between them, which is the key idea behind a knowledge graph.

Let’s take an example sentence: “Rohit Sharma is the captain of the Indian cricket team.”

In this sentence, we can identify several entities:

Rohit Sharma → a person

Captain → role or position

Indian cricket team → organization

As humans, we can understand this sentence easily. But to make machines understand it efficiently, it’s better to represent the data differently — this is where knowledge graphs come in.

A knowledge graph is made up of three main components:

Nodes – represent objects, people, places, or nouns. In our example, Rohit Sharma and Indian cricket team are nodes.

Edges – define the relationships between nodes.

Labels – describe the type of relationship. In our example, the label connecting Rohit Sharma to the Indian cricket team is “captain”.

Let’s consider another sentence: “Virat Kohli is a player of the Indian cricket team.”

Here, Virat Kohli is another node, and the relationship to the Indian cricket team is labeled “player”.

Now, if we combine these relationships in a graph, we can see that Rohit Sharma and Virat Kohli are indirectly connected through the Indian cricket team. This is the power of a knowledge graph — it can help us derive connections between entities and reveal hidden relationships.

Large-scale knowledge graphs are used in applications like Google Search. For instance, if you search for Virat Kohli, Google also shows related players like Rohit Sharma. This happens because all the entities and their relationships are stored and connected in a knowledge graph.

Let’s see some real examples:

Searching Krishna on Google shows details like YouTube channels, social profiles, and books — all connected through a knowledge graph.

Searching Albert Einstein shows education, research, family connections, and related scientists like Isaac Newton or Stephen Hawking. These suggestions are derived from the knowledge graph connecting all relevant nodes and relationships.

Knowledge graphs don’t just power search engines. They also influence YouTube search, ad recommendations, and many other AI-driven applications.

In the next part of this series, we will discuss Neo4j’s graph database, which allows us to create, store, and query knowledge graphs efficiently.

For example, in Neo4j:

Nodes represent entities

Relationships (edges) connect nodes

Property keys store additional information

Neo4j uses Cypher queries to retrieve and manipulate data. For instance, you can query relationships like “Chief Executive Officer”, and Neo4j will show that Elon Musk is the CEO of Tesla. You can also query best-selling Tesla models or even insert creative content like poems and define relationships between them.

When combined with LangChain, Neo4j allows automatic creation of nodes and relationships, making it easy to build applications with rich, interconnected data.

This concludes our introduction to knowledge graphs. In the upcoming videos, we will dive deeper into Neo4j and Cypher queries, and start building interesting graph-based applications from scratch.

I hope you found this explanation easy to understand. See you in the next video. Thank you!

**C) Creating Your Neo4j AuraDB Database Instance**

So guys, in this section of the video, we are going to start with creating our Neo4j’s graph database account.

Here, we will specifically use Neo4j Aura DB. If you search for “Neo4j Aura DB,” you will find the link to create your own database on the cloud platform. The best thing about Neo4j Aura DB is that you can run a single instance and develop multiple projects through it.

Once you click on the link, you will see the Neo4j Aura DB page. It is described as the world’s leading graph database, fully managed as a cloud service — zero admin, globally available, and always on.

To get started, click on Start Free. You can continue signing up with Google or any email ID. After agreeing to the Terms of Service, you will see three plans: Free, Professional, and Enterprise.

If you are just trying it out for learning or small development projects, the Free plan is perfect. It does not require a credit card and provides access to all graph tools. The free plan is limited to 200,000 nodes and 400,000 relationships, which is sufficient for learning purposes.

Once you select the free plan, a password is automatically generated. Make sure to download and save these details. You will get the instance URI, username, password, instance ID, and instance name, which are important when you start coding, especially with LangChain later.

After this, you can categorize your project as personal or work-related, and specify the use case, like Data Science or Data Engineering. Then, your first instance will be created automatically.

Creating the instance may take a few minutes. Once completed, you can see your instance along with details like the number of nodes and relationships created. The cloud region for your instance is displayed, and you will also see the connection URI.

To access your database, click Open, enter your database username and password, and click Connect. You can save this information so you can quickly open the instance in the future.

At this point, your instance is ready, but you have not created any nodes, relationships, or property keys yet. You can start writing queries in the Neo4j interface to create nodes, relationships, and property keys.

Please follow all these steps carefully and have your account ready. In the next section of the videos, we will discuss how Neo4j is different from MySQL and explore its unique capabilities.

So yes, make sure your account and instance are ready, and I will see you all in the next section.

**D) RDBMS VS Graph Database**

So in this section we are going to discuss about the basic differences between RDBMS versus graph database.

When we talk about RDBMS this is nothing but this is my relational relational DBMs right database management system. Along with this we are going to compare with the graph database. If I probably take some of the example of RDBMS, it is nothing but like MySQL SQL server, right? I hope everybody may have known about this. Right. And you may have also used it.

So in order to compare the differences. So let me go ahead and write RDBMS over here versus our graph database.

So the basic difference is that I really want to talk about and this is important before we start, you know uh creating our entire nodes properties in graph database okay.

So first is related to tables right. So in r d uh RDBMS uh we basically create tables. Right. Similarly in graph database here instead of creating tables, we will be creating something called as graphs.

The second important difference is that inside this table we insert records right. We insert records. And this records will be something called as rows right. In the case of graph database we specifically use something called as properties and its it's values. Right? So here, instead of saying it as zeros we specifically say it as properties.

The third thing over here and again we will be discussing about all these things when we do it in our practical implementation. If I probably talk about tables rows are there. Rows are for rows records right. Whenever we insert a record. But in every row there will be some kind of columns. And with respect to this columns, there will be data also. Right. Some data will definitely be there.

So similarly over here, uh, sorry. Uh, this rows uh, I just need to make some change over here. When we say rows right over here inside this graph database we have something called as nodes. Okay. Nodes. Now this nodes will be like this circular structure. Right? It can be an entity. It can be a name. Something like that.

When we talk about columns and data these are nothing. But it is basically properties and values. Properties and values. So how we represent columns and data in RDBMS. Similarly in graph database we represent it as properties and values.

Then uh coming to the fourth and the important point which is called as constraints. I hope you have heard about this kind of constraints, where you have something like primary key, where you have foreign key where you have candidate key, right? So different types of keys are there. And the reason of making keys is that there will be a dependency of another table in a specific table itself.

Similarly, over here we if we really want to create constraints in graph database, we basically create relationships. Relationships like one of the example I told. Right. So this is my two nodes. And this may be related to this right. Let's say over here I have Tesla. This is a company which is owned by Elon Musk. Right. So I can just go ahead and write owned. Right. So owned is basically a relationship.

And in tables, uh, most of the queries we write a lot of joins, queries itself. Right now instead of writing join queries in graph database we specifically use something called as traversal. Traversal basically means based on the properties based on the nodes, based on the relationship, we travel from one node to the other node to finally get a result right.

So these are some of the basic differences between RDBMS and graph database, right? Instead of using tables, we use graph over here. Instead of inserting records or rows, we create nodes over here. Then we have columns and data. Instead of columns and data we use properties and values. Similarly, with respect to constraint we have this primary key foreign key candidate key in RDBMS. Similarly here we have relationship and instead of join queries we use something called as traversal.

Now let's talk about some of the important advantages. So let's go ahead and talk about advantages of neo4j's okay. Now there's some very good advantages. First, entire this neo four J database works with respect to a graph data model. We will talk more about this graph data model. Uh this graph data model will be responsible in creating this nodes um relationship properties and values and many more things properties and values.

Okay, now coming to the second and very important point, when we use this neo for Jay in the cloud, it provides us real time insights, like as soon as I create a graph, because see, at the end of the day when we use this neo 4G there, you could see in that particular instance nodes were getting maintained in a separate way. Properties were getting maintained in a separate way. The relationships were getting maintained in a separate way right here.

Uh, you'll be able to see that if I go to the third point, you will be able to retrieve the data very much easily just by writing one query. And over here we will be learning about this, which is called as Cypher query. like how we have a sequel query. Similarly, we'll go ahead and uh, write the cipher query. And sometime you don't even have to write the cipher query, right. Just by clicking on any of the relationships. And all automatically will be getting a basic template of all the queries itself. And that is all possible because of this Neo4j's database.

Okay. As I said, uh, let's talk about this Cypher query. So we basically say Cypher query as this Cypher query language. Cypher query language. Right now inside the Cypher query language. Uh, it is you can just say it as it is a declarative. It is a decorated declarative query language. To represent. The graph. Visually. It is used to represent the graph visually.

Okay. And the best thing about Cypher query will be that you'll be able to understand in a very much easy way, because it is human readable. Okay. And one of the very important thing that usually in RDBMS you sometime when there is a nested queries and all, you really have to use lot of joints. But uh, in neo4j's you don't have to use any joints. That is the best thing.

So that makes this entire Cypher query language amazing for doing multi different different purposes for retrieving different, different kind of data itself. Okay. Uh, so we will be talking more about it as we go ahead. And I've also spoken about the advantages of neo for Jay, but if I probably talk about most of the common properties like acid properties, it supposed, uh, it neo four J supports complete acid properties, right?

When we say acid, what does this basically mean? In RDBMS there are four important properties. One is atomicity, consistency, isolation, and durability. Right. So let me just go ahead and write this. It is nothing but atomicity. Okay. Consistency. Isolation. Along with this you also have this durability. So with respect to this acid properties which is one of the core features of RDBMS. Similarly this is also supported in Neo4j's.

Um and um, you'll be able to see that, uh, once you start using this, there is no such kind of fixed schema. Okay. So one of the very good advantages with respect to Neo4j's is that we don't have any fixed schema. It is having a flexible schema okay. It's not like that one of the node will have only three properties. The other node will also have three properties. No, nothing like that. It can have any number of properties. It can depend right based on the data that we have.

So this was just to give you a clear idea about the differences between RDBMS and graph database. Along with this, we also saw what are the advantages of uh specifically using neo for J. Okay, now, uh, in the next video or in the next section, we are going to discuss about the data model in Neo Forge. And that will actually give us a crux, understanding about how a nodes is basically created. What are the important things? What are the important components. When we go ahead and create this kind of nodes? By using Neo Forge database, by using the Cypher query language, I this was it. I will see you all in the next section.

**E) Neo4j Property Graph Data Model**

So in this section of the video we are going to discuss about Neo4j's property Graph data model.

Now this Neo4j's graph database uses something called as property graph model property graph model. And this model is specifically used to store. And manage data. Okay. So we'll get an idea about it as we go ahead as we proceed okay.

Now this is very much important. What exactly is property graph model. Now this model represents three main important thing. One is nodes. Second one is something called as relationships. And the third is nothing. But let me write it in another color so it looks good.

So the first is nothing. But over here I will go ahead and write. It is nothing but nodes. Notes. Second is relationships. Third is something called as properties.

So this property graph model or this model represents the data, the entire data, whatever data you have right over here in this three important in in in specifically this three important things that is nodes relationships and properties. Properties are just like rows and records. So this will specifically be having key value pairs. Okay. It will be having key value pairs.

And one more thing that you really need to understand when we talk about this relationship. This is both uni directional. And bi directional. So whenever we go ahead and create any kind of relationship, you can actually create it in the form of uni directional. And you can also create it in the form of bi directional okay.

And every relationship okay. Every relationship. See if I probably consider this relationship, let's say this is my node one. This is my node two. Right. So let's go ahead and write node one. Node two okay.

Now with respect to this node one and node two, whenever we create any relationship let's say this is having a relationship of this. Let's say node one is the parent. Or I can go ahead and write node one is the parent of node two. Right now in this relationship. This node one is just like your start node. And if I probably consider this node two, it is nothing but it is my two node. Or it can we can also say two or uh end node. Okay.

Since we are basically going to write this right. See this is this is also called a start node. It can also be called as something called as from node okay. This can also be called as two node. So this is really important for you all to understand okay.

But if I probably consider the entire graph DB data model. So if I probably consider graph DB data model. Okay. Here there are three important blocks okay. One is nodes. One is relationship. And the third one is something called as properties.

Okay. As I said, uh, I've given you multiple examples over here. So let's say this is one of my node. This is my another node. This is my third node. Okay.

Here, uh, I will go ahead and take one movie. Right. So let's say Avenger okay. So let's say Avenger and Tony Stark okay. So I will say hey Tony Stark. Tony Stark was an actor in Avenger, right? So this is one kind of relationship. Okay.

Um, like, Tony Stark was an actor in Avenger. They can be another movie where Tony Stark was again an actor. Right here. I guess he was a villain. I'm just putting. Okay. There was a villain relationship with respect to one of the movie. This movie can be any other movie, right? Movie two.

So this way you'll be able to see at any point of time this pink color node. This pink color circle is basically called as node. This is called as relationship. Relationship. Right. This actor is basically a kind of uh, you can just consider that it is a kind of property, but we can define this property in key value pairs also, which I will probably be showing you. Right.

So, uh, this is what is entirely, uh, you know, the understanding of the graph, uh, Neo4j's property graph data model. And again, uh, I said you right. This can be bidirectional. Also it can be a relationship. It can be unidirectional. It can be bidirectional. Okay.

So like this, uh, if we have an entire NLP text data, right. Just imagine an NLP text data then just by using. And this internally uses something called as graph neural network. With the help of this graph neural network we will be able to create this right now from where we will be getting this graph neural network. For this we will be using LangChain.

Okay. Um again, LangChain has a good graph DB functionalities which will be able to take that particular text data. And uh, we will be able to convert this entire into this particular nodes. Okay.

But for that again we right. We need to write Cypher query. But if you just directly want to convert it with the help of model we can use a graph neural network. And I think graph neural network uh functionalities this also you can probably go ahead and train it from completely scratch.

But right now we'll focus more on to the LangChain, uh, which has features to convert this particular data insert into a graph DB database itself.

So yeah. Uh, this was all about graph data model. Now, uh, in the next section of the video, we will go ahead and do some variety of practicals and we'll try to insert some data in our Neo4j's RDB. So yes, this was it. I will see you all in the next video.

**F) Getting Started With Cypher Query Language**

So in this section of the video we are going to probably go ahead and create this nodes, relationships, properties, labels. You know and we'll try to see that how this entire information is basically created with the help of Cypher Query language okay. Very important video altogether because this will actually help you to understand how to write a specific query.

So here, uh, first of all, you as soon as you open your Neo4j's AuraDB, once you go inside this particular instance, you'll be able to see this particular text box. And this is where you write your own query right now, let us go ahead and try some of the queries over here, and we'll see that how we can go ahead and create nodes, relationship property keys and many more things okay.

So let's go ahead. So first of all what I'm actually going to do I'm going to go ahead and use this create keyword. And let's create our first node okay. So I will just go ahead and write node Node one okay.

So once I create this create node one. So it is just going to create my first node. And once I execute this here you will be able to see that hey node one is basically created. Okay.

But I'm not able to see this node one right. If I go ahead and click this it shows hey I've assigned some ID over here. This ID value is zero.

Okay. So I told you that whenever we try to create nodes, nodes are nothing but entities. Name something like that. Right. And obviously over here I did not provide any kind of information over here when I probably created this particular node, because I need to provide some kind of labels. Right.

If I say, hey, I'm going to create a node one, let's say node one is something over here. And if I give a name like Krishna, then this particular node name will be assigned to Krishna. Right.

So let me just go ahead and, in order to return everything, you can see match n that is nothing but nodes return n limit 25.

Okay. So now let me just go ahead and create another node. So I will go ahead and write create node. And you can create multiple nodes also. But here what I'm actually going to do I will just go ahead and create one of the nodes. And along with that I will just go ahead and assign some kind of labels.

Okay. Now when I say I'll be assigning some kind of labels, what does this basically mean? So let's say I will go ahead and write Ellen okay Elon Musk okay. And this is my node that I'm actually creating. And let's say I'll go ahead and write he's the CEO right.

So once I execute this you'll be able to see now I'm able to see one of the nodes which is called as CEO. Okay. My node is Ellen. But over here I have tried to assign some kind of label value which is called as CEO. Okay. Otherwise I could have created another node like famous personality and I could assign one of the label like Ellen. Right. So this way I can now see my nodes over here.

Okay, so if I really want to go ahead and see, like who is the CEO? So if I just go ahead and create over here, here you'll be able to see this particular information. Right. And once I go ahead and click this okay. This shows the node details. This is my CEO. The key ID value is one.

Okay. So till here I hope you got an idea of how to properly create a node. Right. And here you can see as soon as I clicked on CEO it gave me a result where match n colon Co return n limit 25 basically means 25 results is allowed to retrieve. And here I can probably see one of them.

Okay. So I hope you got an idea with respect to creating a simple way of this kind of nodes. Okay, now let me do one thing. Let me just go ahead and create one more type here. I will go ahead and write Ellen okay. Ellen I will create this particular node here. I will go ahead and say he is also a CEO right of Tesla. Along with that he is also an employee. So I'm going to create this now.

See here what exactly the CEO and employee will basically do. CEO one is the label over here okay. And here I'm trying to add some more labels. Right. Like employee also one more label that I really want to create. So if I go ahead and execute this here, you can see that now it has got executed and my one more node has actually got created. It is nothing but employee.

Right. And if I go ahead and click on employee. So this is my second node right. Still CEO and employee. There is no relationship. So we are not able to probably see the entire relationship of this specific node.

Okay. But here, if I write this query match n colon employees return n limit 25. Okay. So this is one of the query with respect to Neo4j's. So all these things we have actually tried okay. And this is amazing because now we have also understood that fine we can create multiple labels okay. Just with respect to one node. So I hope you have very much clear with this basic things Right.

Now similarly, I can create different different nodes with a lot of different information. Okay. Let's say I go ahead and write create Ellen okay. And inside this I go ahead and write okay fine. He is the CEO okay.

Now with respect to the CEO, I can go ahead and write some more information. Now I know that this is my label, okay. But being a CEO, this person has some kind of name. Elon has name. Right. So I will go ahead and write my name called as Elon Musk.

So this is nothing but this is called as properties okay. So here I'm trying to create a node with this label name and this property. So I'll say hey Elon Musk over here. I'll say okay let's go ahead and write this year of birth, let's say I will go ahead and write in 1979. Okay, I hope so. Uh, he was born in 1979 or some other date, but you can go ahead and validate it.

Okay. And then let's say the place of birth I have heard is from South Africa. Okay. So I'll just go ahead and write SC okay. Okay. Perfect. So here you can see that I have mentioned all this information. And this is my property keys key value pairs. These are my properties for this particular label okay.

Or I can also go ahead and probably inverse this entire information. But right now we are able to to share or store this particular name and all. So now let's go ahead and execute this. Once I executed this you will be able to see. Now I have this property keys name okay. Name node is nothing but Elon Musk, POB over here you can see node is South Africa and yob is nothing but the node information is 1979.

So this information you are able to see that. Yeah. Now we have also created some kind of property keys. Right. And this is how the query is just there.

See if I really want to understand what is the value for this name. So it is automatically creating this particular query match n where n dot name is not null Return distinct. Return distinct node as entity and dot name as limit 25 Union all match this particular parameters right R dot name is not null return distinct relationship as entity R dot name as name limit 25.

You'll be seeing that this all kind of queries will try to generate it from LangChain. So here the main idea is just to make you understand how quickly you can probably create your nodes, your relationships and property keys right now.

I hope you are able to understand this now again, in order to get all the information out there, I will just go ahead and write match and okay, so let's go ahead and write match end. And here I'm just going to return end. Let me go ahead and write this match end. And here I'm just going to go ahead and return N okay.

So if I go ahead and execute this I will be able to see okay fine. These are all my nodes that you can see over here. If I go ahead and click on Elon Musk now you can see right. So Elon Musk is one of the name over here. Right. And you are able to see all these values. And this node details is nothing but CEO right.

So here you can see POB, name Elon Musk, year of birth. Every information is probably visible over here. Right. And this is amazing because here I made sure that I created my based on the name. You can see the node name is basically assigned. If you are not providing let's say if you have just provided label initially, the label will go away. But if you provided some property key value pairs where you have name information that it has replaced over here, right.

If you see another node two is here, one is over here. This is my SEO details. And the other nodes itself. Right. So blank by blank you will be able to see this. Yes I am able to see this. Circular nodes each and everywhere. Okay. Perfect.

So I hope uh, you are able to understand this entire information. Right. And, uh, you are able to see that how we have easily created this entire nodes. Okay.

Now let's go ahead and try to create relationships. Okay. Now you know that Elon Musk is the CEO of Tesla.

Okay. Now if I really want to create first of all I need to create two entities. Okay. So let's go ahead and create this. So here I will write okay I have already created Elon Musk okay. Now let me just go ahead and write my the Tesla company details.

So here I will write re create. So I will just go ahead and write. Create. So this will be create. Now inside create I will go ahead and write okay. Let's say this is Tesla okay I will provide uh label over here called as company okay.

And with respect to this particular company, let me just go ahead and give some information in the form of key value pairs. So I'll say name. Name is nothing but Tesla. Okay. So here you can see Tesla is the name of the company. And this is the label. So if I go ahead and execute it you'll be able to see I'm able to get the company. So if I go ahead and just click this I'm able to see Tesla. Okay. Perfect.

Now I want to create a relationship between Elon Musk and obviously, uh, if I probably say Elon Musk with Tesla right now here, you will be able to see that if I click on name, right, Elon Musk is there and Tesla is there. Right.

So, uh, how do I probably go ahead and do this? So if I go ahead and click on company. So this is company. This is CEO okay. And CEO these are my three information right. So Elon Musk. So let's let's see how we can probably join this okay. Or how we can go ahead and create this relationship already if you know in the downwards direction.

If I go right here you will be able to see where I have actually used Elon. Right. So here I've created a node called as Elon. Right now what I will write since there is already a node with the name Elon. So if I go ahead and press this, I think uh, it was this one itself. Right. So let's see this.

The node is there. So here what I will say I will create create Ellen okay. And then I will put a relationship. In order to put a relationship I have to use Dash I will use a are parameter. And here I will say CEO. And here I'm just going to give this arrow direction and let me go ahead and write my another node.

Another node will be what my another node will be specifically for this particular company. Right. And the company name that I have actually created, a company node that I've actually created is what Tesla. Right. So here what I'm actually going to write quickly, I will say hey go ahead and create this relationship with Tesla.

Now once I execute this you'll be able to see created two nodes, created one relationship. Now CEO is one of the relationship. Now if I go ahead and click this see the magic okay. So here you can see five is basically getting created. Six is basically created. Now what I have actually done is that I used this particular node Elon with Tesla.

And I don't know why Elon uh, when I see down okay, I had actually created this entire thing. So spelling is wrong or correct. Let's see Elon with CEO right. So here 00000.

So if I go ahead and say Elon with CEO of Tesla it created two nodes. but it is giving me this particular node name, right. The reason is very simple, because this fifth and sixth is actually creating as a new node. Okay. It did not use the older nodes itself. Okay.

So in order to do this let let's take one more very good example. Okay. I will just create it from scratch and then we'll get to know why that node name did not come okay.

Now once I have created my entrepreneur node Krish and my country node India, I can see that Krish lives in India. By double-clicking on the nodes, I can see the relationship “lives in” connecting them. Similarly, clicking on India shows all the properties of that node. This demonstrates how relationships can easily be visualized in Neo4j.

When working with a large database, it’s important to maintain consistent labels. For example, if I want to create a node for a person, I should use the label “Person” consistently across all such nodes. So I created a node labeled Person with the name “Tom Hanks” and birth year 1956.

Next, I created a node for a movie with the label Movie and title “Forrest Gump,” released in 1994. I used placeholders like P for Person and M for Movie so that I can reuse these patterns for multiple records with different properties. This way, only the values like name or title change, keeping the structure consistent.

To create a relationship between Tom Hanks and Forrest Gump, I first used a MATCH query to locate the specific nodes. I matched the Person node with name “Tom Hanks” and the Movie node with title “Forrest Gump.” Then, I created a relationship called acted_in from the Person node to the Movie node. Executing this query successfully created the relationship, showing that Tom Hanks acted in Forrest Gump.

Sometimes, while creating relationships, warnings like “Cartesian product between disconnected patterns” may appear. This happens when nodes are not properly matched or there are multiple nodes without a clear connection. Despite the warning, the relationship can still be successfully created, as demonstrated with Tom Hanks acting in Forrest Gump.

Similarly, I created another Person node, “Crush,” born in 1989, and linked it to the same movie Forrest Gump using the acted_in relationship. Now, both Tom Hanks and Crush are connected to the movie node, showing how multiple relationships can be represented in Neo4j.

To retrieve nodes and relationships, the MATCH keyword is used. For example, MATCH (p:Person {name: "Tom Hanks"}) RETURN p retrieves the Tom Hanks node. Using a relationship pattern, such as MATCH (p)-[:acted_in]->(m) RETURN p, m, allows retrieving both the nodes and the relationships between them. This is useful for exploring the graph and visualizing connections.

Updating node properties is also straightforward. For example, if Tom Hanks’ birth year needs to be corrected from 1956 to 1957, I can use the MATCH query to locate the node and the SET keyword to update the property: MATCH (p:Person {name: "Tom Hanks"}) SET p.yob = 1957 RETURN p. Executing this query updates the property, and the change is visible when inspecting the node.

Throughout this process, writing queries step by step helps in understanding how nodes, relationships, and properties are created, updated, and retrieved in Neo4j. By practicing these queries, users gain familiarity with Cypher and can effectively manage graph data.

In the next session, we will explore more advanced queries, including complex traversals, filtering by properties, and performing multiple operations in a single query. This will further enhance our ability to work with Neo4j and fully leverage the power of graph databases.

This concludes the demonstration of creating nodes, relationships, property keys, and basic Cypher queries. The next video will build upon these concepts for more advanced graph operations.

**G) Intermediate To Advance Cypher Query Language**

We are going to continue our discussion on graph databases and explore some intermediate to advanced examples. In this session, we will see how to upload real datasets into a Neo4j graph database using CSV files. I have created a GitHub link where I have uploaded several CSV files, including movies.csv, posts.csv, ratings.csv, and others. We will use these CSV files to populate our graph database and discuss the queries required to do this.

The use case we will focus on is a social media platform. In this platform, users can create posts, like posts, and have friends. The CSV files contain information about users, their posts, and friendships. For example, user_social.csv contains user_id, name, age, and city. The friendship mapping file contains user_id1, user_id2, and the type of friendship. For instance, 1 to 2 indicates that John and Sarah are friends. Finally, post.csv contains post_id, content, user_id, and timestamp for each post.

To load users from the CSV, we use the built-in LOAD CSV function. We start by writing LOAD CSV WITH HEADERS FROM '<CSV_URL>' AS row to read the file, and then for each row, we create a User node using CREATE (u:User { user_id: toInteger(row.user_id), name: row.name, age: toInteger(row.age), city: row.city }). If we need to delete previous nodes before reloading, we can execute MATCH (n:User) DELETE n. After executing this, all users such as Sarah, John, Mike, David, and Linda will appear as nodes in the database, each with the correct properties.

Next, we load posts and map them to their respective users. We begin with LOAD CSV WITH HEADERS FROM '<POST_CSV_URL>' AS row to load the CSV file. Then we match the user by user_id using MATCH (u:User {user_id: toInteger(row.user_id)}) and create a relationship to a Post node with CREATE (u)-[:POSTED]->(p:Post { post_id: toInteger(row.post_id), content: row.content, timestamp: datetime(row.timestamp) }). Executing this query links each post to the correct user, and we can visualize posts such as “Love New York” or “Beautiful day in Chicago” associated with their authors.

To load friendships, we first read the relationships CSV with LOAD CSV WITH HEADERS FROM '<RELATIONSHIP_CSV_URL>' AS row. Then we match both users using MATCH (u1:User {user_id: toInteger(row.user_id1)}), (u2:User {user_id: toInteger(row.user_id2)}) and create the friendship relationship with CREATE (u1)-[:FRIEND]->(u2). After executing this query, the friend network is established. For example, John is friends with Sarah, Mike, David, and Linda, and Sarah is friends with Mike, creating a complete friend circle.

Once the data is loaded, we can start retrieving information using Cypher queries. To retrieve all users, we execute MATCH (u:User) RETURN u, and to get all posts, we write MATCH (p:Post) RETURN p. If we want to find the friends of a specific user like John, we use MATCH (u:User {name: "John"})-[:FRIEND]->(f:User) RETURN f.name. To retrieve posts made by the friends of John, the query becomes MATCH (u:User {name: "John"})-[:FRIEND]->(f:User)-[:POSTED]->(p:Post) RETURN f.name, p.content. We can also count the number of friends each user has with MATCH (u:User)-[:FRIEND]->(f:User) RETURN u.name, count(f) AS number_of_friends ORDER BY number_of_friends DESC. This will show users like David and Sarah with five friends each, and others with four.

Additionally, we can explore likes on posts. To find users who liked a specific post by John, we match the user and their post with MATCH (u:User {name: "John"})-[:POSTED]->(p:Post)<-[:LIKES]-(l:User) RETURN l.name, p.content. This query traverses the LIKES relationship and shows who liked John’s posts.

Throughout this session, we have seen how to load users, posts, and friendships from CSV files, create nodes and relationships, and retrieve meaningful information using Cypher queries. This demonstrates the power of graph databases for representing complex relationships. The skills learned here are especially useful for advanced applications, such as building generative AI systems with LangChain, where we need to traverse and manipulate graph data efficiently.

# **XXVI) Practical Implementation With Graphdb With Langchain**

**A) Getting Started-Creating Environment**

So we are going to continue our discussion on working with graph databases with LangChain. In this session, we will implement some exciting projects using Neo4j AuraDB along with Python programming, demonstrating how to create generative AI applications. Before we start, it is important to set up the development environment properly and ensure all necessary libraries are installed.

First, we will create a Python environment. Open your command prompt and run "conda create -p ./VENV python==3.12 -y". This will create a new environment named VENV with Python 3.12. Once the environment is created, activate it using "conda activate ./VENV" and clear the terminal to keep things tidy.

Next, we need to prepare a requirements.txt file for our project. Since we are working with Python 3.12 and Neo4j, it is important to use compatible versions of libraries. The requirements.txt should include the following:

"neo4j==5.11.0"
"langchain-community"
"python-dotenv"
"openai"


If you plan to use open-source LLM models, you can also include "langchain_grok". To install all the packages in your environment, run "pip install -r requirements.txt".

After installing the libraries, it is crucial to set up a .env file to store sensitive information such as API keys. For example, if you are using the Grok LLM, your .env file should contain: "GROK_API_KEY=your_grok_api_key_here". This allows your Python scripts to securely access the API without hardcoding sensitive information.

Once the environment is set up and the .env file is ready, we can proceed to connect to Neo4j AuraDB. Make sure you have your database credentials saved, as they will be needed to insert data and execute queries. Proper setup ensures that the Python scripts, LangChain, and Neo4j can work together seamlessly for creating generative AI applications.

**B) Inserting Data In Graph DB With Python And Langchain**

So we are going to continue our discussion with respect to building generative AI projects using GraphDB. In this session, we will see how to create a project that leverages LangChain along with a graph database to respond to user queries intelligently. The main plan here is simple: when a user submits a query, it first goes to our LLM (Language Model), which will generate a corresponding Cypher query. For example, if a user asks, "Who acted in Toy Story 2?", the LLM will create a Cypher query to fetch the relevant data from our graph database. Once the graph DB executes the query, the response is returned to the LLM, which formats it into an output response for the user. This block, where we integrate the LLM, Cypher query, and GraphDB, essentially forms our graph agent, which can answer questions intelligently by querying the graph database.

To start, we create a folder for our project and add a Jupyter Notebook file called "experiments.ipynb". First, we set up the kernel, and then we configure environment variables to store essential database credentials such as Neo4j’s URI, username, and password. This can be done using Python with "import os" and then setting variables like "os.environ['NEO4J_URI'] = 'your_uri'", "os.environ['NEO4J_USERNAME'] = 'your_username'", and "os.environ['NEO4J_PASSWORD'] = 'your_password'". These values can also be read from a .env file if preferred. Ensure that "ipykernel" is installed via "pip install ipykernel" to use Jupyter Notebook properly.

Next, we import the Neo4j graph interface from LangChain using "from langchain_community.graphs import Neo4jGraph" and initialize the graph connection with "graph = Neo4jGraph(url=Neo4j_URI, username=Neo4j_username, password=Neo4j_password)". This object allows us to interact with our Neo4j AuraDB instance.

For the dataset, we are using a movies dataset available in CSV format on GitHub. It contains columns like movie ID, release date, title, actors, directors, genres, and IMDb ratings. To load this dataset into the graph, we define a Cypher query inside Python as a multi-line string. The query starts with "LOAD CSV WITH HEADERS FROM 'your_csv_url' AS row" and uses MERGE statements to create nodes for movies, persons (actors and directors), and genres. Individual properties like movie ID, release date, title, and IMDb ratings are set using "M.id = row.movieID", "M.released = date(row.release)", "M.title = row.title", and "M.imdb_rating = toFloat(row.idb_rating)". For actors and directors, a "FOREACH" loop is used: "FOREACH director IN split(row.directors, '|') | MERGE (p:Person {name: trim(director)}) MERGE (p)-[:DIRECTED]->(M)". Similarly, for actors: "FOREACH actor IN split(row.actors, '|') | MERGE (p:Person {name: trim(actor)}) MERGE (p)-[:ACTED_IN]->(M)". Genres are also handled with: "FOREACH genre IN split(row.genres, '|') | MERGE (g:Genre {name: trim(genre)}) MERGE (M)-[:HAS_GENRE]->(g)".

Once the query is fully defined, we execute it using "graph.query(movie_query)". After execution, refreshing the schema with "graph.refresh_schema()" and "print(graph.schema())" will show all nodes and relationships that were created. In Neo4j, this will result in a graph where movies are linked to actors and directors, and movies are also categorized by genres. For example, "Toy Story 2" will have nodes for Tim Allen and Tom Hanks linked with ACTED_IN, and directors like John Lasseter linked with DIRECTED. Genres like Comedy, Fantasy, and Animation will also be connected to the corresponding movie node.

This setup demonstrates the fundamental process of ingesting a CSV dataset into a graph database using LangChain and Cypher queries. In the next step, we will use the LLM to generate queries dynamically and retrieve information interactively from the graph database, allowing us to build a full question-answer system over GraphDB.

This is how we can integrate LangChain, Neo4j, and Python to create a generative AI application capable of querying and reasoning over graph data.

**C) Creating GraphQuery Chain With Langchain**

So we are going to continue our discussion with respect to generative AI with GraphDB and LangChain. In our previous session, we learned how to insert data into Neo4j. Now, we will see how to leverage an LLM model to query the graph database dynamically. First, we need to set up our environment to use the Grok API. This can be done by writing "import os" and "from dotenv import load_dotenv", and then initializing it with "load_dotenv()". Next, we fetch our API key using "grok_api_key = os.getenv('GROK_API_KEY')". To load the LLM model, we import "from langchain_grok import ChatGrok" and create our LM instance with "lm_model = ChatGrok(grok_api_key=grok_api_key, model_name='gamma')"; here, "gamma" refers to the new open-source model.

Once the LM model is initialized, we can set up a chain to handle generating Cypher queries for the graph database. We import the necessary chain using "from langchain.chains import GraphCypherQAChain" and initialize it with "chain = GraphCypherQAChain.from_llm(llm=lm_model, graph=graph, verbose=True)". The "graph" parameter refers to our previously created Neo4jGraph instance, and "verbose=True" ensures we can see the step-by-step processing. This chain internally uses a prompt template where the input variable is question and the schema is passed from the graph database. The task is automatically defined as generating Cypher statements to query the graph while only using the relationships and properties defined in the schema.

Now, we can use this chain to process queries dynamically. For example, if we want to find the director of the movie Casino, we write "response = chain.run('Who was the director of the movie Casino?')" and execute it. The chain generates the corresponding Cypher query, which might look like "MATCH (m:Movie {title: 'Casino'})<-[:DIRECTED]-(p:Person) RETURN p.name", executes it on the graph, and returns the answer "Martin Scorsese". Similarly, if we want the full cast, we can query with "response = chain.run('Who were the actors in the movie Casino?')" and the model returns "Robert De Niro, Joe Pesci, Sharon Stone, James Woods", dynamically generating the Cypher query and executing it.

This approach demonstrates how we can combine an LLM model with a graph database using LangChain’s "GraphCypherQAChain". By simply initializing the LM model and passing it along with the graph to the chain, we can query the database in natural language and get accurate responses. This makes interacting with complex graph datasets very intuitive, as the LLM automatically generates valid Cypher queries based on the schema. The full process involves: importing necessary modules, initializing the Grok API key, creating the LM model, initializing the GraphCypherQAChain with the graph and LLM, and then running queries with "chain.run()" to get results.

This setup highlights the power of combining LangChain, Neo4j, and open-source LLMs, enabling us to ask natural language questions and receive precise answers without manually writing Cypher queries. It’s a simple yet effective way to build an intelligent question-answer system over a graph database.

**D) Prompting Statergies GraphDB With LLM**

So we are going to continue our discussion on graph databases with Liang Chen. In the previous session, we learned how to quickly generate a simple Q&A system where we could chat with the graph database. The basic idea was simple: the LLM was generating Cypher queries based on user input, executing them on the graph database, and returning the results.

However, there are scenarios where the LLM may not perform perfectly, especially with complex queries. To improve the accuracy of graph database query generation, we will now explore prompting strategies that help the LLM understand the database context better. These strategies focus on providing relevant, database-specific information in the prompts.

First, we will create a new notebook called prompt_strategies.ipynb and set up our environment as usual using import os and load_dotenv(). We will also ensure that our graph database is initialized and data is loaded so that queries can be executed successfully.

Next, we include all our LLM models so that we can access them for generating queries. After that, we create the GraphCypherQAChain using the LLM and graph instance. Here, we can also use parameters like exclude_types to ignore certain fields (for example, genre) and verbose=True to track execution steps.

With the chain ready, we can focus on prompting strategies. The idea is to provide few-shot examples that map natural language questions to Cypher queries. For instance:

Question: “How many artists are there?” → Query: MATCH (a:Person)-[:ACTED_IN]->(m:Movie) RETURN count(DISTINCT a)

Question: “Which actors played in the movie Casino?” → Query: MATCH (a:Person)-[:ACTED_IN]->(m:Movie {title:'Casino'}) RETURN a.name

Question: “How many movies has Tom Hanks acted in?” → Query: MATCH (a:Person {name:'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)

These examples form the basis of our few-shot prompt template. Using LangChain’s FewShotPromptTemplate, we define:

Prefix: Explains to the LLM that it is a Neo4j expert and should generate syntactically accurate Cypher queries.

Suffix: Provides the user’s question and expects the corresponding Cypher query as output.

Examples: Top N examples of natural language questions mapped to Cypher queries.

Input Variables: Includes question (user input) and schema (database schema).

Once the prompt template is created, we integrate it into our GraphCypherQAChain by passing it as cypher_prompt=few_shot_prompt. This helps the LLM understand the pattern of questions and how to generate accurate queries.

Now, we can execute queries such as:

“How many artists are there?” → Returns count of actors.

“Which actors played in the movie Casino?” → Returns actor names.

“How many movies has Tom Hanks acted in?” → Returns the correct count.

Some limitations remain—for example, complex queries like “Actors who acted in multiple movies” may generate invalid Cypher statements. This highlights that while prompting strategies improve LLM performance, the model can still make mistakes, especially with very complex queries.

Overall, by using few-shot prompting strategies, we guide the LLM to generate more accurate Cypher queries and improve the quality of the results from the graph database. This approach gives a clear structure for combining LLMs with Neo4j for natural language Q&A, while also illustrating potential pitfalls to watch out for.

This concludes the discussion on prompting strategies for graph databases. You can now experiment with your own queries and examples to see how well the LLM generates Cypher queries.

# **XXVII) Detailed Intuition and Implementation Of Finetuning LLM Models**

**A) What Is Quantization Indepth Intuition**

In this video, we are going to talk about quantization. Quantization is a concept related to how data is stored in memory, specifically for language models (LMs) or deep learning neural networks such as Transformers or BERT. When discussing quantization in LM models, we are mainly referring to the weights and parameters of the neural network. These weights are usually represented as matrices, and the standard format for storing them is in 32-bit floating point numbers (FP32), also called full precision. FP32 stores numbers in memory with 1 bit for the sign, 8 bits for the exponent, and 23 bits for the mantissa (fractional part).

Quantization essentially means converting data from a higher memory format to a lower memory format. For example, converting FP32 (32-bit) weights into FP16 (16-bit) or int8 (8-bit integers). This process is crucial for managing large models that may have tens of billions of parameters, like LLaMA 2 with 70 billion parameters. Directly loading such models into a standard GPU with limited VRAM or system RAM is often impossible without consuming massive cloud resources, which can be costly. By quantizing the model, we reduce the memory requirements and make inference faster and more feasible on limited hardware. This is especially important when deploying models on mobile phones, edge devices, or smartwatches, where memory and computation are limited.

Quantization can be full precision (FP32) or half precision (FP16). FP16 reduces the memory usage by allocating fewer bits for the exponent and mantissa, while still representing floating point numbers. Similarly, int8 quantization converts values into 8-bit integers, which allows for faster computation and smaller model sizes, but may introduce some loss of accuracy due to reduced precision. To mitigate this, we use a process called calibration, which ensures the conversion from higher precision to lower precision minimizes error. Calibration involves finding the right scale and zero-point for the weights so that the distribution of the numbers fits into the lower-bit representation.

Quantization can be implemented mathematically using formulas. For symmetric unsigned int8 quantization, we can map a range of floating point numbers to 0–255. For example, if weights are in the range 0–1000 and we want to convert them to 0–255, the scale factor can be calculated as scale = (x_max - x_min) / (q_max - q_min), where x_max = 1000, x_min = 0, q_max = 255, and q_min = 0. Any number x can then be quantized using the formula q_x = round(x / scale). For instance, if x = 250, then q_x = round(250 / 3.92) results in 64. This is a straightforward way to convert FP32 weights to 8-bit integers while keeping the distribution intact.

For asymmetric quantization, the data distribution may not be centered around zero, such as when weights range from -20 to 1000. In this case, we compute a scale factor as (x_max - x_min) / (q_max - q_min) = (1000 - (-20)) / 255 ≈ 4.0. A zero-point adjustment ensures that the minimum floating point value (-20) maps correctly to 0 in int8 representation. Here, zero_point = round(-x_min / scale) = 5, so the quantized value of -20 becomes 0 after applying this offset. These two parameters, scale and zero-point, are crucial for accurate quantization.

The process of squeezing high precision values into lower-bit representations is also referred to as calibration. Calibration ensures that the quantized model maintains as much accuracy as possible, even after converting from FP32 to lower-bit formats. In practice, quantization can be applied using two primary modes: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

Post-Training Quantization is applied to a pre-trained model without changing the original weights. The model weights are calibrated and converted into lower-bit formats to create a quantized model suitable for inference. PTQ is simpler but may result in a small loss of accuracy, especially for large and sensitive models.

On the other hand, Quantization-Aware Training (QAT), also called Q8, involves incorporating quantization into the training loop. Here, the trained model is quantized, but we also fine-tune it with additional training data. This helps mitigate the accuracy loss that occurs due to quantization. QAT is especially important for fine-tuning tasks, where maintaining model performance is critical. Python frameworks like TensorFlow or PyTorch allow implementing QAT with minimal code while applying calibration and training logic. For instance, in TensorFlow, you might write: "tf.quantization.quantize(model, input, scale, zero_point)" to perform quantization with calibration.

Finally, two additional concepts that are essential for fine-tuning are LoRA (Low-Rank Adaptation) and ChLoRA. These techniques allow for efficient parameter updates during fine-tuning on large models without retraining all the parameters from scratch. Combining QAT with LoRA enables efficient deployment of large models in production, especially when working with memory-constrained environments.

In summary, quantization is a method to reduce memory usage and improve inference speed by converting high-precision weights (FP32) to lower precision (FP16 or int8), with calibration ensuring minimal loss of accuracy. PTQ is suitable for inference with pre-trained models, whereas QAT is used for fine-tuning while preserving accuracy. These techniques are widely applied in NLP and computer vision models, particularly when deploying on mobile and edge devices. Calibration, scale, zero-point, and understanding symmetric vs asymmetric quantization are all key concepts for applying quantization effectively.

**B) LORA And QLORA Indepth Mathematical Intuition**

We are continuing with the part two of the fine-tuning series, focusing on LoRA and LoRA in depth with mathematical intuition. LoRA, which stands for Low-Rank Adaptation of large language models, is designed to make fine-tuning large pre-trained models efficient by reducing the number of parameters that need to be updated. Imagine a pre-trained model such as GPT-4 or GPT-4 Turbo that has been trained on massive datasets from the internet, books, and other sources. These models, referred to as base models, have billions of parameters and can predict the next token based on the context of prior tokens. Fine-tuning these models can be done in several ways: full parameter fine-tuning, domain-specific fine-tuning, and task-specific fine-tuning.

Full parameter fine-tuning involves updating all weights in the model, which can be extremely resource-intensive for large models. For instance, updating a GPT-4 model with 175 billion parameters would require huge GPU memory and computational power. This can make downstream tasks like model inference and monitoring difficult. Downstream tasks can include Q&A chatbots, domain-specific applications in finance, sales, or retail, or even text-to-SQL models. Full parameter fine-tuning becomes impractical due to hardware constraints, and this is where LoRA provides an elegant solution by reducing the number of trainable parameters during fine-tuning.

LoRA works by tracking the changes in weights caused by fine-tuning rather than updating all model weights directly. Suppose W_0 represents the pre-trained weights of a model. LoRA introduces two matrices A and B so that the change in weights, ΔW, is represented as a low-rank decomposition: ΔW = B @ A. The updated weights then become W = W_0 + ΔW. This decomposition significantly reduces the number of trainable parameters because instead of storing the full matrix, only the two smaller matrices are stored. For example, a 3x3 weight matrix can be decomposed into a 3x1 and a 1x3 matrix. Conceptually in Python, this can be implemented as: "import torch\nW0 = torch.randn(3,3)\nA = torch.randn(1,3)\nB = torch.randn(3,1)\ndelta_W = B @ A\nW = W0 + delta_W\nprint(W)". This method is scalable even for models with billions of parameters because the rank of the decomposition is much smaller than the original matrix size.

As the rank increases, the number of parameters also increases slightly, but it remains much smaller than updating the full model. Typically, ranks between 1 and 8 are chosen to balance expressivity and memory efficiency. Higher ranks can be used if the model needs to learn more complex patterns. LoRA 2.0 extends this concept by applying the low-rank decomposition systematically to attention matrices like query (Q), key (K), and value (V). For example, a 7B parameter model may only require ~167k trainable parameters using rank=1. This ensures that even with a fraction of the parameters updated, the fine-tuned model retains the expressivity of the base model.

Clara, or Quantized LoRA, combines LoRA with quantization to further reduce memory usage. Quantization reduces the precision of the stored weights, for instance, converting float16 weights to 4-bit representations. During inference, these weights can be used directly or converted back to higher precision if needed. In Python, a simple quantization example can be written as: "W_quant = torch.round(W * 16) / 16\nprint(W_quant)". By combining LoRA with quantization, Clara allows fine-tuning very large models on limited GPU resources. It reduces memory requirements, accelerates training, and makes domain-specific or task-specific fine-tuning feasible.

Overall, LoRA and Clara enable efficient fine-tuning of large language models without updating all parameters. The low-rank decomposition tracks weight updates in smaller matrices, reducing the number of trainable parameters, while quantization reduces memory usage further. This approach allows downstream applications like chatbots, document Q&A, or text-to-SQL models to be developed efficiently. For any model, you can apply LoRA with different ranks based on complexity requirements, and combine it with quantization to optimize memory and computation. Using these techniques, fine-tuning becomes more resource-friendly while retaining performance and expressivity of the base model.

**C) Practrical Implementation fine Tuning Custom Data With Google Gemma Model**

In this tutorial, we are going to fine-tune the Gamma model using the LoRA technique in Keras. Gamma is a fully open-source large language model released by Google, available in multiple sizes such as 2B and 7B parameters. Fine-tuning involves updating the model weights so that it can perform better on a specific domain or dataset. Instead of updating all the billions of parameters, LoRA (Low-Rank Adaptation) allows us to train only a small subset of parameters, significantly reducing memory usage and computation. The first step is to set up the environment, which includes getting the necessary API keys. You need to go to AI Studio at Google and generate an API key. In Python, you can set it as: "import os\nos.environ['GOOGLE_API_KEY'] = 'your_api_key_here'". If you are also using Kaggle datasets, you will need a Kaggle username and key, which can be stored similarly: "os.environ['KAGGLE_USERNAME'] = 'your_username'\nos.environ['KAGGLE_KEY'] = 'your_key'".

Next, we install Keras NLP and ensure we have Keras version >=3. The backend can be selected as Jax, TensorFlow, or PyTorch, and in this tutorial, we are using Jax. To avoid memory fragmentation on Jax, we can configure the memory fraction as: "import jax\njax.config.update('jax_platform_name', 'gpu')\nimport jax.numpy as jnp\nfrom jax import random\nimport os\nos.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'". Once the environment is ready, we import Keras NLP modules: "import keras_nlp\nfrom keras_nlp.models import gamma_causal_lm".

The dataset used for fine-tuning is a JSON Lines (JSONL) file containing instructions and corresponding responses. Each JSON object has two keys: "instruction" and "context". The instruction is the question or prompt, and context is the expected response. We can load the JSONL data into Python as follows: "import json\ndata = []\nwith open('dolly_15k.jsonl') as f:\n for line in f:\n entry = json.loads(line)\n if 'context' in entry:\n data.append({'instruction': entry['instruction'], 'response': entry['context']})".

Once the data is ready, we load the Gamma model. Keras NLP provides a simple interface to load pre-trained models: "model = keras_nlp.models.gamma_causal_lm.from_preset('gamma-2b')" for the 2B parameter model. The tokenizer, padding masks, and token IDs are automatically set up. You can inspect the model summary to see the number of layers and parameters. Before fine-tuning, we can generate predictions using the model with a sampler: "sampler = keras_nlp.samplers.TopKSampler(k=5)\nmodel.compile(sampler=sampler)\nresponse = model.generate("Explain photosynthesis to a child")".

Now, we enable LoRA to reduce the number of trainable parameters. For instance, setting a rank of 4 allows us to reduce billions of parameters to just around 1.3 million trainable ones: "model.enable_lora(rank=4)". We then define the optimizer and loss function for training: "from keras.optimizers import AdamW\noptimizer = AdamW(learning_rate=5e-4, weight_decay=0.01)\nmodel.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])". Here, we also exclude layer norm and bias terms from weight decay for stability. The training process is performed using: "model.fit(data, epochs=1, batch_size=1)". For larger datasets, you can increase epochs and batch size to improve performance.

After fine-tuning, the model is capable of providing responses tailored to the dataset. For example, before fine-tuning, asking "What should I do on a trip to Europe?" might return generic answers, but after fine-tuning, it provides more context-aware responses based on the data in dolly_15k.jsonl. Similarly, questions like "Explain photosynthesis to a child" will now yield more refined explanations. To generate outputs after fine-tuning, the code is the same: "sampler = keras_nlp.samplers.TopKSampler(k=5)\nmodel.compile(sampler=sampler)\nresponse = model.generate('What should I do on a trip to Europe?')".

In conclusion, fine-tuning a Gamma model with LoRA in Keras allows us to adapt large pre-trained models efficiently with minimal GPU memory usage. The key steps involve setting up API keys, preparing the dataset in JSONL format, loading the model, enabling LoRA, configuring the optimizer, training the model, and finally generating predictions. LoRA ensures that only a small fraction of parameters are updated, making it practical to fine-tune even multi-billion parameter models on accessible hardware like Google Colab Pro.

# **XXVIII) End To End finetuning LLM Models With Lamini Platform**

**A) End To End Finetuning LLM Models With Lamini AI Cloud**

Welcome back to the fine-tuning series. In this video, I want to introduce you to an amazing platform called Laminae. This platform is designed to simplify almost everything from using basic language models to creating chatbots, all the way to fine-tuning custom language models with your own data. Unlike the Google Gamma model, which involves a lot of manual setup and configuration, Laminae provides a much easier workflow. The platform even offers a free tier, which allows you to perform limited fine-tuning, run inference, and experiment with top open-source models such as LLaMA 3 and Mistral.

Once you log in to Laminae at "app.laminae.ai", you will get redirected to the dashboard where you can see your remaining inference requests and the API token assigned to you. This API token is important because it will be used to authenticate your requests when interacting with the platform. You can copy the API token and store it in a variable in your code like "laminae.API_key = 'your_api_token_here'". This will ensure that all your calls to Laminae are authenticated and linked to your account.

Before we start fine-tuning, you need to prepare your dataset. Laminae expects the data in a simple JSON format where each example is a dictionary with an "input" key and an "output" key. For instance, your dataset can look like this: "data = [{'input': 'What is Laminae?', 'output': 'Laminae is a platform for training and fine-tuning language models.'}, {'input': 'How do I fine-tune a model?', 'output': 'You can upload your dataset and call the tune function with your API token.'}]". This format makes it easy for the platform to understand your training data and map questions to responses. You can create your dataset manually or generate it from existing Q&A resources.

Once the data is ready, we can start coding the fine-tuning process. First, import the Laminae library using "import laminae" and then instantiate the Laminae class with "LM = laminae.Laminae()". Set your API key with "laminae.API_key = 'your_api_token_here'" and then assign your dataset to a variable like "data = get_data()" if you are using a separate function to load the data. Fine-tuning is initiated with the command "LM.tune(data=data)", which is very simple compared to manually configuring models and optimizers. This single function call uploads your dataset to Laminae, starts the fine-tuning process in the cloud, and takes care of all the necessary backend optimizations including quantization and LoRA adaptation.

You can also customize hyperparameters for your fine-tuning session. For example, if you want to specify the learning rate, you can create a dictionary "fine_tune_args = {'learning_rate': 1.4e-4}" and pass it to the tune function like "LM.tune(data=data, fine_tune_args=fine_tune_args)". Laminae also allows you to set parameters such as maximum training steps, early stopping, and optimizer type if needed. However, even with the default settings, you can achieve a functional fine-tuned model. Once the tuning starts, Laminae provides a URL where you can track training progress, view logs, and monitor metrics such as loss and evaluation results.

After the fine-tuning completes, you can evaluate the model directly in Laminae. Before fine-tuning, if you ask a question like "Can Laminae be used on a regular computer?", the response from the base model might be very generic or incomplete. But after fine-tuning, the same question yields a detailed answer customized according to your dataset. For example, "LM.generate('Can Laminae be used on a regular computer?')" will now return a response enriched by the examples you provided in the dataset. Similarly, all other questions in your dataset will generate more accurate and relevant answers.

Finally, Laminae provides a playground to test your fine-tuned models interactively. You can type any query, ask multiple questions, and immediately see how the model responds. Once you are satisfied, you can create a public link for your model using the platform, which allows anyone to use your fine-tuned model in chatbots or other applications. The best part is that Laminae handles all the heavy lifting behind the scenes, including GPU usage, LoRA, quantization, and model optimization, so you can focus entirely on your data and hyperparameters.

Overall, this approach using Laminae is extremely straightforward. You only need to install the required libraries with "pip install -r requirements.txt", prepare your dataset in input-output format, set your API key, and call "LM.tune(data=data)" with optional hyperparameters. Within minutes, you have a fully fine-tuned language model running in the cloud that can generate accurate, customized responses based on your training data.

# **XXIX) Building Stateful, Multi-Actor Applications Using LangGraph**

**A) Introduction To Langgraph**

Welcome to this exciting new module on building generative AI applications using LangGraph. LangGraph is a powerful library available within LongChain that allows you to create stateful multi-actor applications. Essentially, it enables the creation of agents and multi-agent workflows using large language models (LLMs). One of the most impressive aspects of LangGraph is that these agents can communicate with each other, making it ideal for complex workflows. For instance, in a typical data science project, you might have data analysts, data scientists, product managers, and program managers collaborating. Each role has specific responsibilities, and communication between them is key. LangGraph allows us to replicate this real-world workflow by creating AI agents that can interact and perform distinct tasks automatically.

For example, if you want to create a generative AI application with multiple agents, you could have a data analyst agent, a data scientist agent, and a program manager agent. Each agent would perform its respective task and communicate with others as necessary. LangGraph is designed for efficient stateful management, so the state of your workflow is preserved across different operations. Compared to other LLM frameworks, LangGraph offers unique advantages such as cycle controllability and persistence. This allows you to define workflows that include cycles, which are essential for most agent-based architectures, distinguishing it from DAG-based solutions.

Currently, LangGraph also provides a visual interface called LangGraph Studio, which allows you to create workflows using a drag-and-drop interface. At the moment, this studio is available only for Mac, but once it is released for Windows and Linux, it will make workflow creation even easier. However, for now, we focus on writing Python code to implement workflows, which is actually very beneficial for learning the underlying mechanisms. Once you understand the code, using the visual studio becomes effortless. You can see in practice how a workflow executes step by step—when you ask a question, the request is routed to the appropriate agent, the agent takes an action, executes necessary code, and finally returns the response.

To begin with, we will use Python and the LangGraph library. For instance, if you want to create a simple chatbot with multiple agents, you could start by defining agents and their relationships as nodes in a graph. These nodes are similar to what you would see in a graph database, with edges representing communication paths between agents. Here’s a basic example of initializing LangGraph in Python:

"import langgraph
from langgraph import Graph

Initialize the graph

graph = Graph()

Define agents

graph.add_agent('data_analyst')
graph.add_agent('data_scientist')
graph.add_agent('program_manager')

Define relationships

graph.add_edge('data_analyst', 'data_scientist')
graph.add_edge('data_scientist', 'program_manager')

Run workflow

graph.run('Who created you?')"

In this example, the workflow is executed step by step. When a question is posed, it travels through the graph to the appropriate agent. Each agent takes its action and produces a result, which is then returned. This structured approach allows you to simulate real-world collaboration between different roles. You can track the workflow, monitor how agents interact, and visualize the nodes and relationships, making debugging and enhancement easier.

In the upcoming videos, we will dive deeper into LangGraph and build practical projects to illustrate its power. Initially, we will work on Google Colab to make coding, execution, and testing straightforward. One of the first projects we will develop is a simple chatbot application using LangGraph. As we progress, we will explore more complex multi-agent workflows, stateful management, and interaction between agents. By the end of this module, you will understand how to leverage LangGraph to create sophisticated, multi-agent AI applications, and you’ll also be ready to use LangGraph Studio when it becomes more widely available.

So, this was a brief introduction to LangGraph and its capabilities for building stateful, multi-agent applications with LLMs. Remember that each agent can perform specialized tasks, communicate with other agents, and participate in complex workflows. In the next video, we will start coding with Python in Google Colab, and I will show you exactly how to create a working multi-agent chatbot application.

**B) Creating Chatbots Using Langgraph**

Hello everyone! In this video and the upcoming series, we are going to explore a new module in LangChain called LangGraph. LangGraph is an amazing library that helps you build stateful multi-agent applications with LLMs. It is particularly useful for creating agents and multi-agent workflows. In this series, we will go step by step, and in this first video, we will focus on understanding what LangGraph is, why it is useful, and how to create a simple chatbot using it.

LangGraph allows you to build flows that involve cycles, which is essential for most agentic architectures and differentiates it from DAG-based solutions. It provides controllability and persistence, meaning it maintains agent state and workflow over time. Conceptually, it is similar to a graph database where nodes represent agents and edges represent interactions. This graph-like approach makes it easier to manage complex multi-agent interactions efficiently, especially in scenarios where multiple agents need to collaborate or communicate based on a user query.

The key reasons to use LangGraph are simplification, flexibility, scalability, and fault tolerance. It simplifies development by handling state management and agent coordination automatically. For example, if you have multiple agents performing different tasks—one doing a Google search, another doing a Wikipedia search, and another performing a vector DB search—LangGraph manages the communication and workflow between these agents. Developers can define their own logic and communication protocols for highly customized applications. It allows for large-scale multi-agent applications that can handle high-volume interactions and complex workflows. Additionally, it is fault-tolerant; if one agent encounters an error, the overall workflow continues running without interruption.

To start building a chatbot in LangGraph using Google Colab, first, we install the required libraries using !pip install lang-graph langsmith langchain langchain-grok langchain-community. Next, we set up secret API keys for Grok and LangSmith in Colab. These keys can be accessed using from google.colab import user_data and then grok_api_key = user_data.get("grok_api_key") and langsmith_api_key = user_data.get("langsmith_api_key"). These API keys will be used to interact with LLM models and track the state of interactions.

After setting up the keys, we import the required modules using from typing import Annotated, from typing_extensions import TypedDict, from lang_graph.graph import StateGraph, StartNode, EndNode, from lang_graph.graph.messages import add_messages, and from longchain_grok import ChatGrok. Then we define a state class to manage messages in the chatbot. The class is written as class State(TypedDict): messages: Annotated[list, add_messages]. The messages variable keeps a list of all interactions, and the add_messages function ensures that new messages are appended rather than overwriting previous ones.

Next, we create a graph builder to manage the state and workflow using graph_builder = StateGraph(state_class=State). We then define a ChatBot node function that interacts with the LLM. For example, we can write def chatbot(state): response = LM.invoke(state["messages"]) where LM is our ChatGrok model instance. The returned response is appended to the state messages so that every interaction is tracked. Once the ChatBot function is defined, we add it to the graph using graph_builder.add_node(chatbot) and then connect it with a start and end node using graph_builder.add_edge(StartNode, chatbot) and graph_builder.add_edge(chatbot, EndNode). Finally, we compile the graph with graph_builder.compile() to make it ready for execution.

To display the graph visually, we can use from IPython.display import Image, display and display(graph.get_graph().draw_mermaid_png()). This allows us to see the flow from the start node to the chatbot and then to the end node. Once the graph is compiled, we can interact with the chatbot using a while loop: while True: user_input = input("User: ").lower(); if user_input in ["quit", "q"]: print("Goodbye"); break; for event in graph.stream(messages={"user": user_input}): for value in event.values: print("Assistant:", value["messages"].content). In this loop, the user provides input, the chatbot responds, and all messages are tracked in the state. Typing quit or Q will exit the conversation.

By following these steps, we have successfully created a simple chatbot using LangGraph that manages state, interacts with LLMs, and displays the flow in a graph-based structure. This is just the beginning, and in the upcoming videos, we will build more complex projects using multiple agents and external tools, showcasing the full power of LangGraph in building scalable, flexible, and fault-tolerant multi-agent workflows.

**C) Creating Chatbots With External Tools Workflow With Langraph**

Hello guys! In this video, we are going to develop a multi-agent generative AI application using LangGraph, where we will create a chatbot integrated with third-party tools. The idea is that whenever a user asks a query, the chatbot will first attempt to answer it using LLMs. If the chatbot cannot handle the query, it will then forward the query to an external tool, such as Wikipedia or a research paper API (RCF). This way, we can ensure that the chatbot can handle both simple and complex queries efficiently. The architecture looks like this: the user query goes to the chatbot, the chatbot decides whether to respond itself or delegate to a tool, and then the response is sent back to the user.

To begin, we need to install the required libraries using !pip install lang-graph langsmith langchain langchain-grok langchain-community. The langchain-grok library is specifically used to interact with the Grok API for LLMs. After installing the packages, we import Annotated from typing, TypedDict from typing_extensions, and the necessary modules from LangGraph: StateGraph, StartNode, EndNode, and add_messages from lang_graph.graph.messages. These imports are essential to create the state management class and define the graph for our multi-agent chatbot.

Next, we set up external tools. LangChain provides prebuilt tools like Wikipedia and RCF. We import RCFAPIWrapper and WikipediaAPIWrapper using from langchain_community.utilities.utilities import RCFAPIWrapper, WikipediaAPIWrapper. To execute queries, we import rc_query_run and wikipedia_query_run using from langchain_community.tools import rc_query_run, wikipedia_query_run. We then instantiate these tools: rcf_wrapper = RCFAPIWrapper(top_k_results=1, doc_content_chars_max=300) and wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300), followed by creating the tool objects: rcf_tool = rc_query_run(wrapper=rcf_wrapper) and wiki_tool = wikipedia_query_run(wrapper=wiki_wrapper). This setup ensures that queries sent to these tools return concise results, limited to a specified character count.

Once the tools are ready, we create our state class for the chatbot using class State(TypedDict): messages: Annotated[list, add_messages]. This messages variable keeps a history of all user interactions, and add_messages ensures new messages are appended rather than overwritten. Next, we create a graph_builder with graph_builder = StateGraph(state_class=State) to manage the workflow and state transitions within the chatbot application.

We then set up the LLM. Using Grok, we import the model with from langchain_grok import ChatGrok and initialize it: LLM = ChatGrok(api_key=grok_api_key, model_name="gamma-29b"). Here, grok_api_key is retrieved from Colab secrets using from google.colab import user_data and grok_api_key = user_data.get("grok_api_key"). This LLM instance will handle the responses from the chatbot while being aware of any bound tools. We bind our external tools to the LLM using LLM_with_tools = LLM.bind_tools(tools=[wiki_tool]), giving the model access to Wikipedia for queries it cannot answer itself.

The next step is to define the chatbot function. This function inherits the state and interacts with the LLM: def chatbot(state): return LLM_with_tools.invoke(state["messages"]). Here, state["messages"] contains all previous messages, and the LLM decides whether to respond directly or use a tool. After defining the chatbot, we add it to the graph using graph_builder.add_node(chatbot) and connect it with the start node using graph_builder.add_edge(StartNode, chatbot).

We then create a tool node to integrate third-party tools. We import ToolNode and ToolConditions using from langgraph import ToolNode, ToolConditions and define it as tool_node = ToolNode(tools=[wiki_tool]). We connect the chatbot to the tool node and set conditional edges with graph_builder.add_conditional_edges(chatbot, tool_node, ToolConditions()). This allows the chatbot to decide dynamically whether to interact with the tool or respond directly. Finally, we connect the tool back to the chatbot and the chatbot to the end node using graph_builder.add_edge(tool_node, chatbot) and graph_builder.add_edge(chatbot, EndNode).

After constructing the entire graph, we compile it with graph = graph_builder.compile(). We can visualize the flow using IPython display functions: from IPython.display import display, Image and display(graph.get_graph().draw_mermaid_png()). The flow shows Start → Chatbot → Tool (if needed) → Chatbot → End. Finally, we test the chatbot by streaming user inputs: user_input = "Hi there, my name is John" followed by events = graph.stream(messages={"user": user_input}, stream_mode="values"). We iterate through events and print the responses using for event in events: print(event.values[-1]). When the user input cannot be answered directly, the chatbot automatically queries the Wikipedia tool and returns the result, ensuring robust multi-agent functionality.

This setup creates a full multi-agent chatbot application where the LLM interacts with external tools, dynamically choosing how to respond based on the user query. You can extend this by adding more tools or custom nodes for specialized workflows. This demonstrates how LangGraph simplifies building stateful, tool-augmented chatbots while maintaining a clear and controllable workflow.

**D) End to End Multi AI RAG Chatbots Using Langgraph And AstraDB**

In this video we are going to create an amazing end to end multi AI agent application using land graph and astrolabe.

Now if you don't know about land graph, Landgraaf actually helps you to create AI agents in such a way that you'll be able to manage their state. The communication between the agents. It provides you the entire workflow to probably create this amazing AI applications.

So in this video also, we are going to probably discuss about an end to end project. First of all, we will go ahead and discuss about the architecture, what we are actually trying to implement. And then step by step, we'll go ahead and implement this with the help of land graph.

So here the architecture of the project looks like this. Let's say that I'm trying to create a chatbot application which has this multi AI agent feature which will be able to communicate with each other and it will be able to do multiple task. So whenever a user query comes, there will be a start node in land graph. And this start node will further send it to router. Router basically means whenever I'm probably putting up a user query. We need to route it whether this user query can be fulfilled after requesting from a vector database, or whether it can be fulfilled from an external tool like wiki search.

Now here, just for an example, I'm considering wiki search. But uh, in real world application you can even implement different different external tools you like, right? So wiki search is one. You can also use serp API, you know, to probably do Google search itself. Right. So in this example I'm just going to consider one. Uh, and similarly you can implement with respect to other third party external tools.

Then uh, if I probably consider this side, you will be able to see that it will do a vector DB search. Now this vector db is basically getting created by Astra db. We will try to create this entire vector database in the cloud. Okay. In the cloud. And initially inside this vector database we will be putting some information. The information will be a website page from which we can specifically query right. We will take some URLs and we will read the entire data inside that particular website. And we will store that all in in the form of vectors in the vector database. That is Astra DB. Right. And this will be in the cloud platform. Um, basically it is deployed in the cloud. And I will try to show you the entire configuration and how powerful and amazing Astra DB is.

Okay, after we do this specific thing, then the next step will be that we will get the response from this, and then we can integrate along with LLM along with some prompt engineering like what we really want to do with this particular response. And finally we should be able to get our response. And the final node is basically n. So this is the entire architecture of this project. In terms of line graph. We are going to be we are we are going to use the line graph where we are going to implement this entire thing. Right.

So without wasting any time, let's go ahead and implement this entire thing. Okay. So first of all you really need to know about data stacks. So data stacks actually provides you the access of Astra DB. So in order to access them, uh, you know, you can actually do vector search for real world journey applications. And it is really, really powerful. Okay. And already I have made some videos related to Astra in my previous tutorials also.

So first of all, just go ahead and sign in. So once you probably sign in then you will be able to see this specific page okay. Um if you have not signed in. So first of all you need to sign up over there. Here you will be able to probably create databases. Okay. So in order to create the databases you can actually see over here I will just go ahead and click on Create Serverless database.

Okay. Now there are two options over here. Serverless vector database. Serverless non vector database. But as you all know we are planning to probably create serverless vector database. Okay. And the reason is very simple. Whatever data we have or whatever, uh, whatever data in the form of text we have, we are going to convert that into vector. And we are going to save in this particular vector database itself.

So first of all I'll click this. I will give my vector database name. Let's say this will be test database right. And here I will go ahead and select any of the region. And let me just go ahead and click on Create Database. So as soon as I probably create it it will take some amount of time. So let this get created. Till then I will just wait probably to create this entire database and then we will proceed with the next step.

So yes, let me just go ahead and click this and let me start it. Okay. So this is getting created. It is going to take some amount of time. So once it is getting created I will probably go with the next step. So finally guys here you can see that my database has got created. That is text database a test database. It is a vector database. And please make sure that you keep a note of this database ID, because we will be requiring in order to access this database.

Then the next step will be that we will go ahead and create our tokens. Now in order to create a token, first of all, you need to probably select the role that you are probably going ahead with. So I need administrator user. I will just go ahead and click on Generate Token. Now this token will be important because with the help of this particular token I will go ahead and access my database. Right. So if you also want to know you can also download this JSON file. So let me just go ahead and click and download the JSON file.

Okay. So now I have my access token. Now my database is ready. Now it's time that we go ahead and start our execution of the code. Okay. So, uh, before we go ahead, step by step, I will try to show it to you. Till then, just let me hide my face so that you'll be able to clearly see the code itself. Okay. So here I'm going to use Google Colab. So I will change the runtime. Let me just use any GPU that I want. So I'll just go ahead and click on save. Um, so once I save it, my runtime will be on.

Okay. Now what I'm actually going to do. First of all, we will go ahead and install some libraries. So first of all I require library like lang chain install lang chain. Then we are going to install lang graph. Then along with this we are also going to install cashier okay. So this cashier will be important because I am going to this will be helpful for me to initialize the Astra DB database itself. Right. So I will show you as we go ahead. How why are we using this? Okay.

So here you can see the installation has already taken it. It has already started. Okay. Then the next step after this. So what we are basically going to do is that we are going to import cashier And then we will provide the connection of the of the Astra DB. Okay. So in order to connect to the Astra DB, we will be using this. First of all, the important parameter that I actually require is my Astra DB application underscore token. Okay. Now this particular token we can get it from over here. So here you can see this token information is basically present. So I'm just going to copy this and I'm going to paste it over here.

Okay. So here I'm just going to go ahead and paste it. And with the help of this particular token we will be able to connect to our Astra DB right. Then we are going to probably use our Astra db id. Okay. That is our database ID okay. So we will be using this. We have created this particular variable. Um let's let me go and see this So here is my test database. And here is my database ID. Okay. So this basically becomes my database ID itself right. Now in order to initialize this I require this two information. So I will go ahead and write Cachio dot init which is nothing but sorry dot init and will initialize this particular database.

So token is equal to astra db app store db application token. And the second parameter will basically be my database database id which will be equal to Astra Astra db id okay. So here you can see once I execute this this will get initialized. And here I have got some warnings but it's okay. Now we are able to connect to our Astra DB right. Perfect. Now the next step will be that since we need to we need to develop the land graph itself. First of all, I will also install one more very important thing that is pip install lang chain underscore community. Okay. So let me go to go ahead and install this entire application over here.

Once this is installed uh, I still need to install some more things like uh, lang chain grok, uh, lang chain hub, uh, lang Lang graphs. There are some more libraries that we are just going to install. So for that I will just go ahead and copy and paste over here and I'll talk about what all things we are basically going to install. Okay.

So here you can see I'll be installing lang chain underscore community lang underscore community I think it is already been installed. We will be installing tick token. I'll talk about this. Uh we are going to use open source LM models. So for that we are going to use this Lang chain grok. Then you have lantern hub chrome adb lang chain lang graph lang chain hugging face. I will not require chrome adb because I'm specifically going to use this Datastax DB, right? So line graph line chain hugging face is basically there.

So here you can see I'll be installing lang chain underscore community lang underscore community I think it is already been installed. We will be installing tick token. I'll talk about this. Uh we are going to use open source LM models. So for that we are going to use this Lang chain grok. Then you have lantern hub chrome adb lang chain lang graph lang chain hugging face. I will not require chrome adb because I'm specifically going to use this Datastax DB, right? So line graph line chain hugging face is basically there. So now let me just go ahead and install all.

Once this is done then we are going to probably use our own LLM. So in order to use our own LLM we require the LLM model itself. Now here I'm using sentence transformer model okay. So it is basically embedding model which will convert my text into embeddings. Right. So what I'm going to do is that I will use lang chain sentence transformer which will provide me embedding. And this embedding will be directly stored into our Astra vector database.

Okay. Now once I do that next step will be to fetch some data from a website. So here I will be using web scraper. We will scrape some content from a website. Once we scrape the content we will break the content into chunks. Right. So here I will be using recursive text splitter. You can also use character text splitter. But recursive text splitter works fine because it will split based on paragraphs, sentences, and then we will create embeddings for each of these chunks and store it into vector database.

Once my vector database is ready we are good to go with land graph. So first of all we need to install land graph client. So pip install lang graph client. Once this is installed then we can directly create our graph using python. Right. So here we will create our start node. Start node basically waits for the user input. Then we will create our router node. Router node will decide whether to send this query to vector db search or to external tool.

Now here you can see that I am creating two agents. One agent is vector DB agent. Other agent is external tool agent. Vector DB agent will basically get the response from Astra DB. External tool agent will get response from Wiki search. You can also replace wiki search with any API you want. Right. So this is very modular. You can add multiple external agents as per your requirement.

Now once we have defined our agents we will connect them in a workflow. So start node will connect to router. Router will connect to both vector DB agent and external tool agent. And finally both agents will connect to a final node which will provide response to the user. So this is entire multi AI agent workflow using land graph. You can add more nodes if you want more complexity.

Finally we will test the entire workflow. So once we run our python script it will ask for user query. Let's say I ask “Who is the founder of Tesla?” Router will decide that this query can be fulfilled by external tool agent because this is a general knowledge query. External tool agent will fetch answer from Wiki search and provide response. If I ask “Explain a paragraph from this website” then router will send this query to vector DB agent. Vector DB agent will fetch embeddings from Astra DB and provide response.

So guys this is end to end multi AI agent application using land graph and Astra DB. You can enhance this further by adding multiple LLM models, multiple external tools, and more complex routing logic. This entire setup is modular, scalable, and can be deployed in cloud.
