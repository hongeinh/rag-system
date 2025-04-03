from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

with open("data/penguins.txt", "r") as f:
    text = f.read()

# Load and split text into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
documents = splitter.split_text(text)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_texts(documents, embeddings)

# Set up language model
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2-medium", #facebook/opt-350m, 
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)

# Get rag response
def get_rag_response(query):
    docs = vector_store.similarity_search(query, k=1)
    context = docs[0].page_content

    prompt = f"Based on this context: '{context}', answer the query '{query}' clearly and directly. Answer:"
    response = llm(prompt)
    # print(response)
    return response.split("Answer:")[-1].strip()

if __name__=="__main__":
    print(get_rag_response("What are penguins?"))