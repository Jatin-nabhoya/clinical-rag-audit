# Step 2: Chunk with RecursiveCharacterTextSplitter

from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_text(text: str) -> list[str]:
    """Split a plain text string into token-sized chunks."""
    return splitter.split_text(text)
