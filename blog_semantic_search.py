from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import numpy as np


class SemanticSearch:
    # noinspection SpellCheckingInspection
    def __init__(self, model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        # Initialize the SentenceTransformer model
        self._model = SentenceTransformer(model_name)
        # Placeholder for paragraphs and embeddings
        self._paragraphs = []
        self._embeddings = None

    def load_and_embed_epub(self, epub_file_path):
        # Load EPUB file and convert to paragraphs
        self.__epub_to_paragraphs(epub_file_path)
        # Generate embeddings for paragraphs using the model
        self._embeddings = self.__create_embeddings([para['text'] for para in self._paragraphs])

    def search(self, query, top_results=5):
        # Generate embeddings for the query
        query_embedding = self.__create_embeddings([query])[0]
        # Calculate cosine similarity between query and all embeddings
        scores = self.__cosine_similarity(query_embedding, self._embeddings)
        # Get indices of top results
        results = np.argsort(scores)[::-1][:top_results].tolist()
        return results

    def __epub_to_paragraphs(self, epub_file_path):
        # Load EPUB file
        book = epub.read_epub(epub_file_path)
        # Extract paragraphs from EPUB sections
        for section in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            paragraphs = self.__epub_sections_to_paragraphs(section)
            self._paragraphs.extend(paragraphs)

    @staticmethod
    def __epub_sections_to_paragraphs(section):
        # Convert EPUB section to paragraphs with additional information
        html = BeautifulSoup(section.get_body_content(), 'html.parser')
        p_tag_list = html.find_all('p')
        paragraphs = [{'text': paragraph.get_text().strip(),
                       'chapter_name': ' '.join([heading.get_text().strip() for heading in html.find_all('h1')]),
                       'para_no': para_no}
                      for para_no, paragraph in enumerate(p_tag_list)
                      if len(paragraph.get_text().split()) >= 150
                      ]
        return paragraphs

    def __create_embeddings(self, texts):
        # Generate embeddings for a list of texts using the model
        texts = [text.replace("\n", " ") for text in texts]
        return self._model.encode(texts)

    @staticmethod
    def __cosine_similarity(query_embedding, embeddings):
        # Calculate cosine similarity between query and all embeddings
        dot_products = np.dot(embeddings, query_embedding)
        query_magnitude = np.linalg.norm(query_embedding)
        embeddings_magnitudes = np.linalg.norm(embeddings, axis=1)
        cosine_similarities = dot_products / (query_magnitude * embeddings_magnitudes)
        return cosine_similarities


def test_semantic_search():
    # Instantiate the simplified search class
    simple_search = SemanticSearch()

    # Hardcoded EPUB file path for simplicity
    # noinspection SpellCheckingInspection
    epub_path = (r'D:\Documents\Papers\EPub Books\Karl R. Popper - '
                 r'The Logic of Scientific Discovery-Routledge (2002).epub')

    # Load and embed the EPUB file
    simple_search.load_and_embed_epub(epub_path)

    # Define a query
    query = 'Why do we need to corroborate theories at all?'

    # Perform a search and get the results
    results = simple_search.search(query, top_results=5)

    # Print the results along with the extracted information
    print("Top results:")
    for result in results:
        para_info: dict = simple_search._paragraphs[result]
        chapter_name = para_info['chapter_name']
        para_no = para_info['para_no']
        paragraph_text = para_info['text']
        print(f"Chapter: '{chapter_name}', Passage number: {para_no}, "
              f"Text: '{paragraph_text[:500]}...'")
        print('')


# Example Usage
if __name__ == "__main__":
    test_semantic_search()
