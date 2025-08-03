from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your document
text_loader = TextLoader("./doc/dream.txt")
documents = text_loader.load() 


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)


# Split documents
splits = text_splitter.split_documents(documents)

# Output the results
for i, split in enumerate(splits):
    print(f"Split {i+1}:\n{split}\n")