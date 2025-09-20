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
