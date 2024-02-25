from sentence_transformers import SentenceTransformer
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from os.path import exists
import numpy as np
import math
import os
import unittest
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Optional, Tuple, Union
from typing import TextIO


class SemanticSearch:
    # noinspection SpellCheckingInspection
    def __init__(self,
                 full_file_name: Optional[str] = None,
                 model_name: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                 do_strip: bool = True,
                 min_chapter_size: int = 2000,
                 first_chapter: int = 0,
                 last_chapter: Union[int, float] = math.inf,
                 min_words_per_paragraph: int = 150,
                 max_words_per_paragraph: int = 500,
                 results_file: str = 'results.txt') -> None:
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self._file_name: Optional[str] = full_file_name
        self._do_strip: bool = do_strip
        self._min_chapter_size: int = min_chapter_size
        self._first_chapter: int = first_chapter
        self._last_chapter: Union[int, float] = last_chapter
        self._min_words: int = min_words_per_paragraph
        self._max_words: int = max_words_per_paragraph
        self._results_file: str = results_file
        self._chapters: Optional[List[dict]] = None
        self._embeddings: Optional[np.ndarray] = None
        # Initialize logger
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        if full_file_name is not None:
            self.load_file(full_file_name)

    @staticmethod
    def get_ext(full_file_name: str) -> str:
        return full_file_name[full_file_name.rfind('.'):]

    @staticmethod
    def switch_ext(full_file_name: str, new_ext: str) -> str:
        return full_file_name[:full_file_name.rfind('.')] + new_ext

    @staticmethod
    def read_json(json_path: str) -> Tuple[List[dict], np.ndarray]:
        print('Loading _embeddings from "{}"'.format(json_path))
        with open(json_path, 'r') as f:
            values = json.load(f)
        return values['_chapters'], np.array(values['_embeddings'])

    @staticmethod
    def print_and_write(text: str, f: TextIO) -> None:
        print(text)
        f.write(text + '\n')

    @staticmethod
    def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        # Calculate the dot product between the query and all embeddings
        dot_products = np.dot(embeddings, query_embedding)
        # Calculate the magnitude of the query and all embeddings
        query_magnitude = np.linalg.norm(query_embedding)
        embeddings_magnitudes = np.linalg.norm(embeddings, axis=1)
        # Calculate cosine similarities
        cosine_similarities = dot_products / (query_magnitude * embeddings_magnitudes)
        return cosine_similarities

    @staticmethod
    def fast_cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        # Reshape the query_embedding to a 2D array for compatibility with cosine_similarity
        query_embedding = query_embedding.reshape(1, -1)
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings)
        # Flatten the result to a 1D array
        similarities = similarities.flatten()

        return similarities

    @staticmethod
    def epub_sections_to_chapter(section: epub.EpubHtml) -> Optional[dict]:
        # Convert to HTML and extract paragraphs
        html = BeautifulSoup(section.get_body_content(), 'html.parser')
        p_tag_list = html.find_all('p')
        text_list = [paragraph.get_text().strip() for paragraph in p_tag_list if paragraph.get_text().strip()]
        if len(text_list) == 0:
            return None
        # Extract and process headings
        heading_list = [heading.get_text().strip() for heading in html.find_all('h1')]
        title = ' '.join(heading_list)
        return {'title': title, 'paragraphs': text_list}

    @property
    def do_strip(self) -> bool:
        return self._do_strip

    @do_strip.setter
    def do_strip(self, value: bool) -> None:
        self._do_strip = value

    @property
    def min_chapter_size(self) -> int:
        return self._min_chapter_size

    @min_chapter_size.setter
    def min_chapter_size(self, value: int) -> None:
        self._min_chapter_size = value

    @property
    def first_chapter(self) -> int:
        return self._first_chapter

    @first_chapter.setter
    def first_chapter(self, value: int) -> None:
        self._first_chapter = value

    @property
    def last_chapter(self) -> Union[int, float]:
        return self._last_chapter

    @last_chapter.setter
    def last_chapter(self, value: Union[int, float]) -> None:
        self._last_chapter = value

    @property
    def min_words_per_paragraph(self) -> int:
        return self._min_words

    @min_words_per_paragraph.setter
    def min_words_per_paragraph(self, value: int) -> None:
        self._min_words = value

    @property
    def max_words_per_paragraph(self) -> int:
        return self._max_words

    @max_words_per_paragraph.setter
    def max_words_per_paragraph(self, value: int) -> None:
        self._max_words = value

    def load_model(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def load_file(self, full_file_name: str) -> None:
        # Assert the full file name has an .epub or .json extension
        assert self.get_ext(full_file_name) in ['.epub', '.json'], ('Invalid file format. '
                                                                    'Please upload an epub or json file.')
        self._file_name = full_file_name
        # Load the embeddings
        if self.get_ext(self._file_name) == '.epub':
            # Create a json file with the same name as the epub
            json_file_name = self.switch_ext(self._file_name, '.json')
            # Check if the json file exists
            if not exists(json_file_name):
                # No json file exists, so create embeddings to put into a json file
                self.__embed_epub(self._file_name)
            else:
                self._file_name = json_file_name
        # A json file should now exist with our embeddings
        assert self.get_ext(self._file_name) == '.json', 'Should now be a json file.'
        # Do we now have embeddings and chapters? If not, load them
        if self._chapters is None or self._embeddings is None:
            # Load the embeddings from the json file
            self._chapters, self._embeddings = self.read_json(self._file_name)

    def search(self, query: str, top_results: int = 5) -> Tuple[List[str], List[int]]:
        results_msgs: List[str] = []
        # Create _embeddings for the query
        query_embedding = self.__create_embeddings(query)[0]
        # Calculate the cosine similarity between the query and all _embeddings
        scores = self.fast_cosine_similarity(query_embedding, self._embeddings)
        # Grab the top results
        results = np.argsort(scores)[::-1][:top_results].tolist()
        # Write out the results using the with statement to ensure proper file closure
        with open(self._results_file, 'a') as f:
            file_msg = 'File: "{}"'.format(self._file_name)
            self.print_and_write(file_msg, f)
            query_msg = 'Query: "{}"'.format(query)
            self.print_and_write(query_msg, f)
            for i in results:
                # Convert the index (into a list of flattened paragraphs which is what embeddings is)
                # into a chapter and paragraph number
                paragraph, title, paragraph_num = self.__index_into_chapters(i)
                result_msg = ('\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'
                              .format(title, paragraph_num, scores[i], paragraph))
                results_msgs.append(result_msg)
                self.print_and_write(result_msg, f)
            self.print_and_write('\n', f)
        return results_msgs, results

    def preview_epub(self) -> None:
        epub_file_name = self._file_name
        assert self.get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
        self.__epub_to_chapters(epub_file_name)
        self.__print_previews()

    def __embed_epub(self, epub_file_name: str) -> None:
        # Assert this is an epub file
        assert self.get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
        # Create a json file with the same name as the epub
        json_file_name = self.switch_ext(epub_file_name, '.json')
        # Delete any existing json file with the same name
        if exists(json_file_name):
            os.remove(json_file_name)
        # Create the json file name
        json_file_name = self.switch_ext(epub_file_name, '.json')
        # Convert the epub to html and extract the paragraphs
        self.__epub_to_chapters(epub_file_name)
        print('Generating embeddings for "{}"\n'.format(epub_file_name))
        paragraphs = [paragraph for chapter in self._chapters for paragraph in chapter['paragraphs']]
        # Generate the _embeddings using the _model
        self._embeddings = self.__create_embeddings(paragraphs)
        # Save the chapter _embeddings to a json file
        try:
            print('Writing embeddings for "{}"\n'.format(epub_file_name))
            with open(json_file_name, 'w') as f:
                json.dump({'_chapters': self._chapters, '_embeddings': self._embeddings.tolist()}, f)
            self._file_name = json_file_name
        except IOError as io_error:
            print(f'Failed to save embeddings to "{json_file_name}": {io_error}')
        except json.JSONDecodeError as json_error:
            print(f'Failed to decode JSON in "{json_file_name}": {json_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

    def __index_into_chapters(self, index: int) -> Tuple[str, str, int]:
        flattened_paragraphs = [{'text': paragraph, 'title': chapter['title'], 'para_no': para_no}
                                for chapter in self._chapters
                                for para_no, paragraph in enumerate(chapter['paragraphs'])]

        return (flattened_paragraphs[index]['text'], flattened_paragraphs[index]['title'],
                flattened_paragraphs[index]['para_no']
                if 0 <= index < len(flattened_paragraphs) else None)

    def __epub_to_chapters(self, epub_file_name: str) -> None:
        book = epub.read_epub(epub_file_name)
        item_doc = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        book_sections = list(item_doc)
        chapters = [result for section in book_sections
                    if (result := self.epub_sections_to_chapter(section)) is not None]
        self._chapters = chapters
        if self._do_strip:
            self.__strip_blank_chapters()
        self.__format_paragraphs()

    def __create_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [text.replace("\n", " ") for text in texts]
        return self._model.encode(texts)

    def __format_paragraphs(self) -> None:
        # Split paragraphs that are too long and merge paragraphs that are too short
        for i, chapter in enumerate(self._chapters):
            for j, paragraph in enumerate(chapter['paragraphs']):
                words = paragraph.split()
                if len(words) > self._max_words:
                    # Split the paragraph into two
                    # Insert paragraph with max words in place of the old paragraph
                    maxed_paragraph = ' '.join(words[:self._max_words])
                    chapter['paragraphs'][j] = maxed_paragraph
                    # Insert a new paragraph with the remaining words
                    new_paragraph = ' '.join(words[self._max_words:])
                    chapter['paragraphs'].insert(j + 1, new_paragraph)

                # Merge paragraphs that are too short
                while len(chapter['paragraphs'][j].split()) < self._min_words and j + 1 < len(chapter['paragraphs']):
                    # This paragraph is too short, so merge it with the next one
                    chapter['paragraphs'][j] += '\n' + chapter['paragraphs'][j + 1]
                    # Delete the next paragraph since we just merged it to the previous one
                    del chapter['paragraphs'][j + 1]

            # After the loop, handle the case where the last paragraph is too short
            last_index = len(chapter['paragraphs']) - 1
            last_para_len = len(chapter['paragraphs'][last_index].split())
            prev_para_len = len(chapter['paragraphs'][last_index - 1].split()) if last_index > 0 else 0
            if last_para_len < self._min_words and last_index > 0 and prev_para_len + last_para_len < self._max_words:
                # Merge the last paragraph with the previous one
                chapter['paragraphs'][last_index - 1] += '\n ' + chapter['paragraphs'][last_index]
                # Remove the last paragraph since we just merged it to the previous one
                del chapter['paragraphs'][last_index]

            # Remove empty paragraphs and whitespace
            chapter['paragraphs'] = [para.strip() for para in chapter['paragraphs'] if len(para.strip()) > 0]
            if len(chapter['title']) == 0:
                chapter['title'] = '(Unnamed) Chapter {no}'.format(no=i + 1)

    def __print_previews(self) -> None:
        for (i, chapter) in enumerate(self._chapters):
            title = chapter['title']
            wc = len(' '.join(chapter['paragraphs']).split(' '))
            paras = len(chapter['paragraphs'])
            initial = chapter['paragraphs'][0][:100]
            preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
            print(preview)

    def __strip_blank_chapters(self) -> None:
        # This takes a list of _chapters and removes the ones outside the range [first_chapter, last_chapter]
        # or if the chapter is too small (likely a title page or something)
        last_chapter = min(self._last_chapter, len(self._chapters) - 1)
        chapters = self._chapters[self._first_chapter:last_chapter + 1]
        # Filter out _chapters with small total paragraph size
        chapters = [chapter for chapter in chapters if sum(len(paragraph) for paragraph
                                                           in chapter['paragraphs']) >= self._min_chapter_size]
        self._chapters = chapters


def test_ebook_search(do_preview=False):
    # noinspection SpellCheckingInspection
    book_path = \
        r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).epub'
    # book_path = r"D:\Documents\Papers\EPub Books\KJV.epub"
    ebook_search = SemanticSearch()
    if not do_preview:
        ebook_search.load_file(book_path)
        query = 'Why do we need to corroborate theories at all?'
        results = ebook_search.search(query, top_results=5)
        print(results)
    else:
        ebook_search.preview_epub()


class TestSemanticSearch(unittest.TestCase):

    def setUp(self):
        # Set up any necessary variables or configurations for testing
        # noinspection SpellCheckingInspection
        self._json_path = \
            r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).json'

        # Check if the JSON file exists, if not, load EPUB and save embeddings in JSON
        if not exists(self._json_path):
            # Generate EPUB file path by replacing .json with .epub
            epub_path = self._json_path.replace('.json', '.epub')
            search_instance = SemanticSearch()
            search_instance.load_file(epub_path)

    def test_query(self):
        # Test loading an EPUB file
        search_instance = SemanticSearch()
        search_instance.load_file(self._json_path)

        # Define your query and expected output
        query = 'Why do we need to corroborate theories at all?'
        expected_results = [501, 441, 462, 465, 122]
        expected_results_msgs = '''
            Chapter: "10 CORROBORATION, OR HOW A THEORY STANDS UP TO TESTS", Passage number: 60, Score: 0.69
            "*6 See my Postscript, chapter *ii. In my theory of corroboration—in direct opposition to Keynes’s, 
            Jeffreys’s, and Carnap’s theories of probability—corroboration does not decrease with testability, but 
            tends to increase with it.
            *7 This may also be expressed by the unacceptable rule: ‘Always choose the hypothesis which is most ad hoc!’
            2 Keynes, op. cit., p. 305.
            *8 Carnap, in his Logical Foundations of Probability, 1950, believes in the practical value of predictions; 
            nevertheless, he draws part of the conclusion here mentioned—that we might be content with our basic 
            statements. For he says that theories (he speaks of ‘laws’) are ‘not indispensable’ for science—not even 
            for making predictions: we can manage throughout with singular statements. ‘Nevertheless’, he writes 
            (p. 575) ‘it is expedient, of course, to state universal laws in books on physics, biology, psychology, 
            etc.’ But the question is not one of expediency—it is one of scientific curiosity. Some scientists want 
            to explain the world: their aim is to find satisfactory explanatory theories—well testable, i.e. simple 
            theories—and to test them. (See also appendix *x and section *15 of my Postscript.)"
            '''
        # Call the search method and get the actual results
        actual_results_msgs, actual_results = search_instance.search(query, top_results=5)
        stripped_result_actual = (actual_results_msgs[0].replace(" ", "").replace("\t", "")
                                  .replace("\n", ""))
        stripped_expected = (expected_results_msgs.replace(" ", "").replace("\t", "")
                             .replace("\n", ""))
        self.assertEqual(expected_results, actual_results)
        self.assertEqual(stripped_expected, stripped_result_actual)

    def tearDown(self):
        # Clean up any resources or configurations after testing
        pass


if __name__ == "__main__":
    # test_ebook_search(do_preview=False)
    # unittest.main()
    pass
