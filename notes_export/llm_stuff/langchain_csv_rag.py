"""
LangChain RAG system that loads from a CSV file
"""
import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

def create_rag_from_csv(csv_file_path, source_column=None):
    # Load CSV file
    loader = CSVLoader(
        file_path=csv_file_path,
        source_column=source_column,
        encoding='utf-8'
    )
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents from CSV")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks")
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create custom prompt
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain, vectorstore

def query_rag(qa_chain, question):
    """
    Query the RAG system
    
    Args:
        qa_chain: The QA chain
        question: Question to ask
    """
    result = qa_chain.invoke({"query": question})
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['result']}")
    print(f"\nSource Documents ({len(result['source_documents'])}):")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n--- Source {i} ---")
        print(doc.page_content[:200] + "...")
    
    return result

# Example usage
if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "C:/Users/sberry5/Documents/teaching/UDA/data/all_lyrics_test.csv" 
    
    print("Creating RAG system from CSV...")
    qa_chain, vectorstore = create_rag_from_csv(csv_file)
    
    # Example Queries
    questions = [
        "What information is in this dataset?",
        "Summarize the key points from the data"
    ]
    
    for question in questions:
        query_rag(qa_chain, question)
        print("\n" + "="*80 + "\n")
    
    # Interactive mode
    print("\nEnter 'quit' to exit")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        query_rag(qa_chain, user_question)
