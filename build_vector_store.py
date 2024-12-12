from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema import Document
from llama_index.readers.web import MainContentExtractorReader
import os

# Load the existing vectorstore
persist_directory = "chroma_db_website"

# Initialize Embeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# List of URLs
#urls = ["https://google.com", "https://paulgraham.com/articles.html", "https://www.dawn.com/news/1875445"]
#urls = ["https://www.dawn.com/news/1875445"]
urls = ["https://www.dawn.com/news/1876196/no-slowing-down-psx-as-100-index-continues-bullish-momentum-even-after-surpassing-100000-milestone",
        "https://tribune.com.pk/story/2513208/psx-hits-new-high-as-kse-100-index-crosses-102000-points",
        "https://tribune.com.pk/story/2513117/visa-rejections-for-pakistanis"]


def load_or_parse_data(urls):
    loader = MainContentExtractorReader()
    documents = loader.load_data(urls=urls)
    print(documents)
    return documents

def get_langchain_documents(documents, urls):
    langchain_documents = [
        Document(page_content=doc.text, metadata={"source": url})
        for doc, url in zip(documents, urls)
    ]
    return langchain_documents


# Create vector database
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
def build_database(urls):
    
    vectorstore = get_vector_database()

    # Step 1: Retrieve existing metadata
    existing_data = vectorstore._collection.get()
    existing_urls = {metadata.get("source") for metadata in existing_data['metadatas'] if metadata.get("source")}

    # Step 2: Filter out URLs already in the vector store
    new_urls = [url for url in urls if url not in existing_urls]
    print(f"New URLs to add: {new_urls}")

    if not new_urls:
        print("No new URLs to add.")
        return vectorstore

    # Call the function to either load or parse the data
    documents = load_or_parse_data(new_urls)
    langchain_documents = get_langchain_documents(documents, new_urls)

    print(f"length of documents loaded: {len(langchain_documents)}")

    update_vector_database(vectorstore, langchain_documents)

    return vectorstore


def get_vector_database():
    vs  = create_vector_database(langchain_documents=None)
    return vs



def create_vector_database(langchain_documents):
    vs = None
    if langchain_documents is None:
        # get an empty vector store
           vs = Chroma (
            persist_directory=persist_directory,
            embedding_function=embed_model,
            collection_name="rag"  # Name of your collection
            )
           print('Vector DB retrieved successfully !')
    else:
        # Create and persist a Chroma vector database from the chunked documents
            vs = Chroma.from_documents (
            documents=langchain_documents,
            embedding=embed_model,
            persist_directory=persist_directory,  # Local mode with in-memory storage only
            collection_name="rag"
            )

            print('Vector DB created successfully !')
    return vs


# Create vector database
def update_vector_database(vectorstore, langchain_documents):

    #for doc in langchain_documents:
    #    print (doc.page_content)
    #    print (doc.metadata)
    new_texts = [doc.page_content for doc in langchain_documents]
    new_metadatas = [doc.metadata for doc in langchain_documents]
    # Add new data to the vectorstore
    vectorstore.add_texts(new_texts, metadatas=new_metadatas)

    # Save the updated vectorstore to disk
    vectorstore.persist()


def print_vectorstore(vectorstore):
    # Count the documents in the vectorstore
    print("Total documents in vectorstore:", vectorstore._collection.count())

    # Access the underlying collection and retrieve metadata
    documents = vectorstore._collection.get()

    # Print metadata for each document
    for i, metadata in enumerate(documents['metadatas']):
        print(f"Document {i + 1} Metadata: {metadata}")


    for i, (id, content, metadata) in enumerate(zip(documents['ids'], documents['documents'], documents['metadatas'])):
        print(f"Document {i + 1} Id: {id}")
        #print(f"Document {i + 1} Content: {content}")
        print(f"Document {i + 1} Metadata: {metadata}")
        print("-" * 50)

    #print("Total documents in vectorstore:", vectorstore._collection.metadata)



#vectorstore = build_database(urls)
#print_vectorstore(vectorstore)