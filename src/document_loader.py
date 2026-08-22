from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class DocumentLoader:
    
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.vectorstore: Optional[FAISS] = None
    
    def load_lcel_docs(self, url: str = "https://python.langchain.com/docs/concepts/lcel/") -> Tuple[str, Optional[FAISS]]:
        """
        Load LCEL documentation from the specified URL, split into chunks,
        and build a FAISS vector store for semantic retrieval.
        
        Args:
            url: The URL to load documentation from
            
        Returns:
            Tuple of (concatenated_content, vectorstore)
        """
        print(f"Loading documentation from: {url}")
        print(f"Max depth: {self.max_depth}")
        
        loader = RecursiveUrlLoader(
            url=url, 
            max_depth=2,  # Reduced depth to prevent infinite loops
            use_async=False,
            prevent_outside=True,  # Prevent loading external sites
            link_regex=r".*docs/concepts/lcel.*",  # Only follow LCEL-related links
            extractor=lambda x: Soup(x, "html.parser").text
        )
        
        print("Starting document loading...")
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        
        if not docs:
            return "", None

        # Sort the list based on the URLs and get the text
        d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        
        print(f"Total content length: {len(concatenated_content)} characters")
        
        # Split documents into text chunks for vector indexing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Split documentation into {len(chunks)} chunks")
        
        # Build FAISS vector store
        try:
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            print("Successfully created FAISS vector store")
        except Exception as e:
            print(f"Warning: Could not initialize FAISS vector store: {e}")
            self.vectorstore = None

        return concatenated_content, self.vectorstore
    
    def retrieve_context(self, query: str, k: int = 4) -> str:
        """
        Retrieve relevant documentation chunks for a query using FAISS.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            Formatted string of top k relevant context chunks
        """
        if self.vectorstore is None:
            return ""
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n --- \n\n".join([doc.page_content for doc in docs])
    
    def load_custom_docs(self, urls: List[str]) -> Tuple[str, Optional[FAISS]]:
        """
        Load documentation from multiple URLs and build vector store.
        
        Args:
            urls: List of URLs to load documentation from
            
        Returns:
            Tuple of (concatenated_content, vectorstore)
        """
        all_docs = []
        
        for url in urls:
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=self.max_depth,
                extractor=lambda x: Soup(x, "html.parser").text
            )
            docs = loader.load()
            all_docs.extend(docs)
        
        if not all_docs:
            return "", None

        # Sort and concatenate
        d_sorted = sorted(all_docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(all_docs)
        
        try:
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            print(f"Warning: Could not initialize FAISS vector store: {e}")
            self.vectorstore = None

        return concatenated_content, self.vectorstore

