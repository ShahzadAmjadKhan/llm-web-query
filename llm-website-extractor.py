import os
from llama_index.core import VectorStoreIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document
from llama_index.readers.web import MainContentExtractorReader
from langchain_groq import ChatGroq
import build_vector_store


from langchain_community.chat_models import ChatOllama

GROQ_API_KEY = '<<ADD GROQ API KEY>>'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# List of URLs
#urls = ["https://google.com", "https://paulgraham.com/articles.html", "https://www.dawn.com/news/1875445"]
#urls = ["https://www.dawn.com/news/1875445"]
urls = ["https://www.dawn.com/news/1876196/no-slowing-down-psx-as-100-index-continues-bullish-momentum-even-after-surpassing-100000-milestone",
        "https://tribune.com.pk/story/2513208/psx-hits-new-high-as-kse-100-index-crosses-102000-points"]


def load_or_parse_data():
    loader = MainContentExtractorReader()
    documents = loader.load_data(urls=urls)
    print(documents)
    return documents



# Create vector database
def create_vector_database(persist_directory, embed_model):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    documents = load_or_parse_data()

    # Split loaded documents into chunks 
    ##text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    ##docs = text_splitter.split_documents(documents)

    for doc in documents:
        print(doc.text)
        print(doc.extra_info.get("url"))

    langchain_documents = [
        Document(page_content=doc.text, metadata={"source": url})
        for doc, url in zip(documents, urls)
    ]


    print(f"length of documents loaded: {len(langchain_documents)}")
    #print(f"total number of document chunks generated :{len(docs)}")
    #docs[0]

    # Initialize Embeddings
    #embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents
    vs = Chroma.from_documents(
        documents=langchain_documents,
        embedding=embed_model,
        persist_directory=persist_directory,  # Local mode with in-memory storage only
        collection_name="rag"
    )

    print('Vector DB created successfully !')
    return vs


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
#


prompt = set_custom_prompt()



chat_model = ChatGroq(temperature=0,
                      #model_name="mixtral-8x7b-32768",
                      model_name="llama-3.1-70b-versatile",
                      api_key=GROQ_API_KEY,)


llm_local_ollama = ChatOllama(model="llama3")

"""

# Initialize Embeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

persist_directory = "chroma_db_website"

if not os.path.exists(persist_directory):
    vs  = create_vector_database(persist_directory, embed_model)
else:
    print(f"The persist directory '{persist_directory}' already exists.")


vectorstore = Chroma(embedding_function=embed_model,
                      persist_directory=persist_directory,
                      collection_name="rag")
 
                      
"""

vectorstore = build_vector_store.build_database(urls)

retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

qa = RetrievalQA.from_chain_type(llm=chat_model,
                                #llm=llm_local_ollama,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})


##response = qa.invoke({"query": "what is complete name of article on Coronavirus?"})


##print(response['result'])


##response = qa.invoke({"query": "list all articles about time?"})


##print(response['result'])

# 4. User interaction loop
print("Ask questions about the documents! Type 'exit' to quit.\n")
while True:
    user_input = input("Your question: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = qa.invoke({"query": user_input})
        print(f"Answer: {response['result']}\n")
    except Exception as e:
        print(f"An error occurred: {e}")