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
