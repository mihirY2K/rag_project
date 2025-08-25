from flask import Flask, render_template, request, jsonify 
from werkzeug.utils import secure_filename
import os



app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    from dotenv import load_dotenv
    import fitz
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    import os
    from werkzeug.utils import secure_filename

    # 1. Load API Key
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 2. Get the uploaded file and question
    uploaded_file = request.files.get("file")
    question = request.form.get("text")
    if not uploaded_file or not question:
        return jsonify({"error": "Missing file or question"}), 400

    # 3. Save file temporarily
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    uploaded_file.save(filepath)

    # 4. Extract text from PDF
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        return jsonify({"error": f"PDF reading error: {str(e)}"}), 500

    # 5. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_text(text)

    # 6. Vector store setup
    try:
        embeddings = OpenAIEmbeddings()
        index_name = "rag-first"  # you can make this dynamic if needed
        vectorstore = PineconeVectorStore.from_texts(documents, embeddings, index_name=index_name)
    except Exception as e:
        return jsonify({"error": f"Vector store error: {str(e)}"}), 500

    # 7. RAG chain setup
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """)
    rag_chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

   
    try:
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Model invocation failed: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)

