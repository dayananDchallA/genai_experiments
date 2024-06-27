from flask import Flask, render_template, request
import PyPDF2
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained question-answering model
nlp = pipeline("question-answering")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    # Get the uploaded PDF file
    file = request.files["file"]
    pdf_text = extract_text_from_pdf(file)

    # Preprocess the extracted text
    preprocessed_text = preprocess_text(pdf_text)

    # Get the user's question
    question = request.form["question"]

    # Generate the answer using the question-answering model
    answer = answer_question(preprocessed_text, question)

    return render_template("result.html", answer=answer)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    return text


def preprocess_text(text):
    # Implement any necessary preprocessing steps here
    return text

def answer_question(text, question):
    result = nlp(question=question, context=text)
    return result["answer"]

if __name__ == "__main__":
    app.run(debug=True)