from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import numpy as np


class SemanticSearch:
    def __init__(self, model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        # Initialize the SentenceTransformer model
        self._model = SentenceTransformer(model_name)
        # Placeholder for chapters and embeddings
        self._chapters = []
        self._embeddings = None

    def load_and_embed_epub(self, epub_file_path):
        # Load EPUB file and convert to chapters with paragraphs
        self.__epub_to_chapters(epub_file_path)
        # Extract paragraphs from chapters
        paragraphs = [paragraph for chapter in self._chapters for paragraph in chapter['paragraphs']]
        # Generate embeddings for paragraphs using the model
        self._embeddings = self.__create_embeddings(paragraphs)

    def search(self, query, top_results=5):
        # Generate embeddings for the query
        query_embedding = self.__create_embeddings(query)[0]
        # Calculate cosine similarity between query and all embeddings
        scores = self.__cosine_similarity(query_embedding, self._embeddings)
        # Get indices of top results
        results = np.argsort(scores)[::-1][:top_results].tolist()
        return results

    def __epub_to_chapters(self, epub_file_path):
        # Load EPUB file
        book = epub.read_epub(epub_file_path)
        # Extract paragraphs from EPUB sections
        for section in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            chapter = self.__epub_sections_to_chapter(section)
            if chapter:
                # Filter out short paragraphs
                chapter['paragraphs'] = [para for para in chapter['paragraphs'] if len(para.split()) >= 150]
                if chapter['paragraphs']:
                    self._chapters.append(chapter)

    def __epub_sections_to_chapter(self, section):
        # Convert EPUB section to chapter with paragraphs
        html = BeautifulSoup(section.get_body_content(), 'html.parser')
        p_tag_list = html.find_all('p')
        text_list = [paragraph.get_text().strip() for paragraph in p_tag_list if paragraph.get_text().strip()]
        if len(text_list) == 0:
            return None
        title = ' '.join([heading.get_text().strip() for heading in html.find_all('h1')])
        return {'title': title, 'paragraphs': text_list}

    def __create_embeddings(self, texts):
        # Generate embeddings for a list of texts using the model
        texts = [text.replace("\n", " ") for text in texts]
        return self._model.encode(texts)

    def __cosine_similarity(self, query_embedding, embeddings):
        # Calculate cosine similarity between query and all embeddings
        dot_products = np.dot(embeddings, query_embedding)
        query_magnitude = np.linalg.norm(query_embedding)
        embeddings_magnitudes = np.linalg.norm(embeddings, axis=1)
        cosine_similarities = dot_products / (query_magnitude * embeddings_magnitudes)
        return cosine_similarities

    def index_into_chapters(self, index):
        flattened_paragraphs = [{'text': paragraph, 'title': chapter['title'], 'para_no': para_no}
                                for chapter in self._chapters
                                for para_no, paragraph in enumerate(chapter['paragraphs'])]

        return (flattened_paragraphs[index]['text'], flattened_paragraphs[index]['title'],
                flattened_paragraphs[index]['para_no']
                if 0 <= index < len(flattened_paragraphs) else None)


# Example Usage
if __name__ == "__main__":
    # Instantiate the simplified search class
    simple_search = SemanticSearch()

    # Hardcoded EPUB file path for simplicity
    epub_path = \
        r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).epub'

    # Load and embed the EPUB file
    simple_search.load_and_embed_epub(epub_path)

    # Define a query
    query = 'Why do we need to corroborate theories at all?'

    # Perform a search and get the results
    results = simple_search.search(query, top_results=5)

    # Print the results along with the extracted information
    print("Top results:")
    for result in results:
        text, title, para_no = simple_search.index_into_chapters(result)
        print(f"Chapter: '{title}', Passage number: {para_no}, Text: '{text[:1000]}...'")
        print('')

