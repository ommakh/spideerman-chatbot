import langchain
from langchain.llms import GooglePalm
from langchain.schema import document
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import networkx

api_key = 'Enter the api key'
print("we are good...")

llm = GooglePalm(google_api_key=api_key)


from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "enter the file path name"


def create_vector_db():
    from langchain.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(file_path='webman data.csv', encoding="utf-8" )
    datx = loader.load()


    print(datx)
    vectordb = FAISS.from_documents(documents=datx, embedding=instructor_embeddings)
    vectordb.save_local("enter the file path name")


"""
import requests

# Set the API key
API_KEY = "enter the api key"

# Define the model name that is taken from hugging face
MODEL_NAME = "hkunlp/instructor-large"

# Define the text to embed
TEXT = ["This is the text to embed."]

# Construct the request body
requestBody = {
    "api_key": API_KEY,
    "model": MODEL_NAME,
    "inputs": TEXT
}

# Make the request
response = requests.post("https://api.instruct.huggingface.co/pipeline/feature-extraction/{}/embeddings".format(MODEL_NAME), json=requestBody)

# Check the response status code
if response.status_code == 200:
    # Get the embedding vectors
    embeddings = response.json()["outputs"]["embeddings"][0]

    # Print the embedding vectors
    print(embeddings)
else:
    # Handle the error
    print("Error: {}".format(response.status_code))

"""

def get_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retrival = vectordb.as_retriever(score_threshold=0.1)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
       In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
       If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

       CONTEXT: {context}

       QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retrival,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_chain()
    print(chain(""))
