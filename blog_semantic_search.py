from sentence_transformers import SentenceTransformer
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import numpy as np


def epub_to_paragraphs(epub_file_path, min_words=0):
    paragraphs = []
    book = epub.read_epub(epub_file_path)

    for section in book.get_items_of_type(ITEM_DOCUMENT):
        paragraphs.extend(epub_sections_to_paragraphs(section, min_words=min_words))

    return paragraphs


def epub_sections_to_paragraphs(section, min_words=0):
    html = BeautifulSoup(section.get_body_content(), 'html.parser')
    p_tag_list = html.find_all('p')
    paragraphs = [
        {
            'text': paragraph.get_text().strip(),
            'chapter_name': ' '.join([heading.get_text().strip() for heading in html.find_all('h1')]),
            'para_no': para_no,
        }
        for para_no, paragraph in enumerate(p_tag_list)
        if len(paragraph.get_text().split()) >= min_words
    ]
    return paragraphs


def create_embeddings(texts, model):
    return model.encode([text.replace("\n", " ") for text in texts])


def cosine_similarity(query_embedding, embeddings):
    dot_products = np.dot(embeddings, query_embedding)
    query_magnitude = np.linalg.norm(query_embedding)
    embeddings_magnitudes = np.linalg.norm(embeddings, axis=1)
    cosine_similarities = dot_products / (query_magnitude * embeddings_magnitudes)
    return cosine_similarities


def get_embeddings(model, paragraphs):
    texts = [para['text'] for para in paragraphs]
    return create_embeddings(texts, model)


def semantic_search(model, embeddings, query, top_results=5):
    query_embedding = create_embeddings([query], model)[0]
    scores = cosine_similarity(query_embedding, embeddings)
    results = np.argsort(scores)[::-1][:top_results].tolist()
    return results


def test_semantic_search():
    paragraphs = epub_to_paragraphs(r"D:\Documents\Papers\EPub Books\KJV.epub", min_words=3)
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    embeddings = get_embeddings(model, paragraphs)

    query = 'Did Jesus ever cry?'
    results = semantic_search(model, embeddings, query)

    print("Top results:")
    for result in results:
        para_info = paragraphs[result]
        chapter_name = para_info['chapter_name']
        para_no = para_info['para_no']
        paragraph_text = para_info['text']
        print(f"Chapter: '{chapter_name}', Passage number: {para_no}, Text: '{paragraph_text}'")
        print('')


# Example Usage
if __name__ == "__main__":
    test_semantic_search()
