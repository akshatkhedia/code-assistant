from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
    
    def load_lcel_docs(self, url: str = "https://python.langchain.com/docs/concepts/lcel/") -> str:
        """
        Load LCEL documentation from the specified URL.
        
        Args:
            url: The URL to load documentation from
            
        Returns:
            Concatenated content from all loaded documents
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
        
        # Sort the list based on the URLs and get the text
        d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        
        print(f"Total content length: {len(concatenated_content)} characters")
        return concatenated_content
    
    def load_custom_docs(self, urls: List[str]) -> str:
        """
        Load documentation from multiple URLs.
        
        Args:
            urls: List of URLs to load documentation from
            
        Returns:
            Concatenated content from all loaded documents
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
        
        # Sort and concatenate
        d_sorted = sorted(all_docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        
        return concatenated_content
