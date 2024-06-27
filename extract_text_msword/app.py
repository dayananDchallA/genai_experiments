from spire.doc import *
from spire.doc.common import *

# Create a Document object
document = Document()
# Load a Word document
document.LoadFromFile("C:/Users/DayanandChalla/OneDrive - NGENUX SOLUTIONS PRIVATE LIMITED/PlayGround/rag_code_snippets/extract_text_msword/Sample.docx")

# Extract the text of the document
document_text = document.GetText()

# Write the extracted text into a text file
with open(r"C:\Users\DayanandChalla\OneDrive - NGENUX SOLUTIONS PRIVATE LIMITED\PlayGround\rag_code_snippets\extract_text_msword\output\DocumentText.txt", "w", encoding="utf-8") as file:
    file.write(document_text)

document.Close()