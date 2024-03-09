from sentence_transformers import SentenceTransformer
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from os.path import exists
import numpy as np
import math
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Union
from typing import TextIO
from pypdf import PdfReader
import re
from utilities import replace_ligatures


class SemanticSearch:
    # noinspection SpellCheckingInspection
    def __init__(self,
                 full_file_name: Optional[str] = None,
                 model_name: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                 do_strip: bool = True,
                 epub_min_chapter_size: int = 2000,
                 epub_first_chapter: int = 0,
                 epub_last_chapter: Union[int, float] = math.inf,
                 pdf_min_character_filter: int = 25,
                 by_paragraph: bool = True,
                 combine_paragraphs: bool = True,
                 page_offset: int = 0,
                 min_words_per_paragraph: int = 100,
                 max_words_per_paragraph: int = 500,
                 results_file: str = 'output.txt') -> None:
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self._file_name: Optional[str] = full_file_name
        self._do_strip: bool = do_strip

        # Epub settings
        self._min_chapter_size: int = epub_min_chapter_size
        self._first_chapter: int = epub_first_chapter
        self._last_chapter: Union[int, float] = epub_last_chapter

        # Pdf settings
        # Minimum characters per paragraph. If less, filter out the paragraph (for pdfs to elmiminate headers, etc.)
        self._min_character_filter: int = pdf_min_character_filter

        # Page vs Paragraph settings (applies to PDF and EPUB)
        self._by_paragraph: bool = by_paragraph
        self._combine_paragraphs: bool = combine_paragraphs
        self._page_offset: int = page_offset
        # Minimum words in a paragraph. If less than this, combine with next
        self._min_words: int = min_words_per_paragraph
        self._max_words: int = max_words_per_paragraph

        # Working variables
        self._results_file: str = results_file
        # Epub specific
        self._chapters: Optional[List[dict]] = None
        # Used by both epub and pdf
        self._embeddings: Optional[np.ndarray] = None
        self._flattened_text = None
        self._pages = None

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
        return values['text_list'], np.array(values['embeddings'])

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

    @property
    def min_character_filter(self) -> int:
        return self._min_character_filter

    @min_character_filter.setter
    def min_character_filter(self, value: int) -> None:
        self._min_character_filter = value

    def load_model(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def load_file(self, full_file_name: str) -> None:
        # Assert the full file name has an .epub or .json extension
        assert self.get_ext(full_file_name) in ['.epub', '.json', '.pdf'], ('Invalid file format. '
                                                                            'Please upload an epub or json file.')
        # Reset all the variables on load
        self._chapters = None
        self._embeddings = None
        self._flattened_text = None
        # Load files
        self._file_name = full_file_name
        # Load the embeddings
        if self.get_ext(self._file_name) in ['.pdf', '.epub']:
            # Create a json file with the same name as the pdf
            json_file_name = self.switch_ext(self._file_name, '.json')
            # Check if the json file exists
            if not exists(json_file_name):
                # No json file exists, so create embeddings to put into a json file
                self.__embed_file(self._file_name)
            else:
                self._file_name = json_file_name
        # A json file should now exist with our embeddings
        assert self.get_ext(self._file_name) == '.json', 'Should now be a json file.'
        # Do we now have embeddings and chapters? If not, load them
        if self._chapters is None or self._embeddings is None:
            # Load the embeddings from the json file
            text_list, self._embeddings = self.read_json(self._file_name)
            if text_list and isinstance(text_list[0], dict):
                if 'chapter' in text_list[0] and text_list[0].get('chapter') is None:
                    # This is a pdf file
                    self._flattened_text = text_list
                else:
                    # This is an epub file
                    self._chapters = text_list
            else:
                # This should never happen
                raise ValueError('Invalid json file. Please upload a valid json file.')

    def search(self, query: str, top_results: int = 5) -> Tuple[List[str], List[int]]:
        results_msgs: List[str] = []
        # Create _embeddings for the query
        query_embedding = self.__create_embeddings(query)[0]
        # Calculate the cosine similarity between the query and all _embeddings
        scores = self.fast_cosine_similarity(query_embedding, self._embeddings)
        # Grab the top results
        results = np.argsort(scores)[::-1][:top_results].tolist()
        # Convert the index (into a list of flattened paragraphs which is what embeddings is)
        if self._flattened_text is None and self._chapters is not None:
            self.__chapters_to_flat_text()
        # Write out the results using the with statement to ensure proper file closure
        with open(self._results_file, 'a') as f:
            file_msg = 'File: "{}"'.format(self._file_name)
            self.print_and_write(file_msg, f)
            query_msg = 'Query: "{}"'.format(query)
            self.print_and_write(query_msg, f)
            for i in results:
                # into a chapter and paragraph number
                paragraph, title, paragraph_num, page_num = self.__index_into_text(i)
                if page_num is None:
                    result_msg = ('\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'
                                  .format(title, paragraph_num, scores[i], paragraph))
                else:
                    result_msg = ('\nPage number: {}, Passage number: {}, Score: {:.2f}\n"{}"'
                                  .format(page_num - self._page_offset, paragraph_num, scores[i], paragraph))
                results_msgs.append(result_msg)
                self.print_and_write(result_msg, f)
            self.print_and_write('\n', f)
        return results_msgs, results

    def preview_epub(self) -> None:
        epub_file_name = self._file_name
        assert self.get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
        self.__book_file_to_text_list(epub_file_name)
        self.__print_previews()

    def __embed_file(self, book_file_name: str) -> None:
        # Assert this is an epub file
        assert self.get_ext(book_file_name) in ['.epub', '.pdf'], 'Invalid file format. Please upload an epub file.'
        # Create a json file with the same name as the epub
        json_file_name = self.switch_ext(book_file_name, '.json')
        # Delete any existing json file with the same name
        if exists(json_file_name):
            os.remove(json_file_name)
        # Create the json file name
        json_file_name = self.switch_ext(book_file_name, '.json')
        # Convert the epub to html and extract the paragraphs
        self.__book_file_to_text_list(book_file_name)
        print('Generating embeddings for "{}"\n'.format(book_file_name))
        # Handle epub (chapters) vs pdf (pages only)
        paragraphs = []
        text_list = []
        if self._flattened_text is None and self._chapters is not None:
            paragraphs = [paragraph for chapter in self._chapters for paragraph in chapter['paragraphs']]
            text_list = self._chapters
        elif self._flattened_text is not None:
            paragraphs = [para['text'] for para in self._flattened_text]
            text_list = self._flattened_text
        # Generate the _embeddings using the _model
        self._embeddings = self.__create_embeddings(paragraphs)
        # Save the chapter _embeddings to a json file
        try:
            print('Writing embeddings for "{}"\n'.format(book_file_name))
            with open(json_file_name, 'w') as f:
                json.dump({'text_list': text_list, 'embeddings': self._embeddings.tolist()}, f)
            self._file_name = json_file_name
        except IOError as io_error:
            print(f'Failed to save embeddings to "{json_file_name}": {io_error}')
        except json.JSONDecodeError as json_error:
            print(f'Failed to decode JSON in "{json_file_name}": {json_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

    def __chapters_to_flat_text(self):
        self._flattened_text = [{'text': paragraph, 'title': chapter['title'], 'para_num': para_num,
                                 'page_num': None}
                                for chapter in self._chapters
                                for para_num, paragraph in enumerate(chapter['paragraphs'])]

    def __index_into_text(self, index: int) -> Tuple[str, Optional[str], int, Optional[int]]:
        if self._flattened_text is None:
            self.__chapters_to_flat_text()

        return (self._flattened_text[index]['text'], self._flattened_text[index]['title'],
                self._flattened_text[index]['para_num'], self._flattened_text[index]['page_num']
                if 0 <= index < len(self._flattened_text) else None)

    def __book_file_to_text_list(self, file_path: str) -> None:
        file_ext = self.get_ext(file_path)
        if file_ext == '.epub':
            self.__epub_to_chapters(file_path)
        elif file_ext == '.pdf':
            self.__pdf_to_pages(file_path)
            if self._by_paragraph:
                self.__pages_to_paragraphs()
            else:
                self.__cleanup_pages()
        else:
            raise ValueError('Invalid file format. Please upload an epub or pdf file.')

    def __epub_to_chapters(self, epub_file_name: str) -> None:
        book = epub.read_epub(epub_file_name)
        item_doc = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        book_sections = list(item_doc)
        chapters = [result for section in book_sections
                    if (result := self.epub_sections_to_chapter(section)) is not None]
        self._chapters = chapters
        if self._do_strip:
            self.__strip_blank_chapters()
        if self._combine_paragraphs:
            self.__format_chapter_paragraphs()

    def __pdf_to_pages(self, pdf_file_path: str) -> None:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            pages = []
            for page_num in range(len(pdf_reader.pages)):
                # pdf_page_text_test = pdf_reader.pages[page_num].extract_text()\
                #     .encode('utf-8', 'replace').decode('ascii', 'replace')
                pdf_page_text = pdf_reader.pages[page_num].extract_text().strip()
                # original_text = pdf_page_text
                # Replace non-ASCII characters with empty strings
                # pdf_page_text = re.sub(r'[^\x00-\x7F]+', ' ', pdf_page_text)
                # pdf_page_text = re.sub(r'[^\x00-\x7F‘’]+', ' ', pdf_page_text)
                # Replace ligatures with their non-ligature equivalents
                pdf_page_text = replace_ligatures(pdf_page_text)
                # Remove letter-hyphen-space and letter-hyphen-letter to remove hyphen
                pdf_page_text = re.sub(r'(?<=[a-zA-Z])-\s|(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])',
                                       '', pdf_page_text)
                # Replace if there is a missing space after a period, add it
                pdf_page_text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1. \2', pdf_page_text)
                # Fix single quotes to not have spaces next to them
                pdf_page_text = re.sub(r'‘ ', '‘', pdf_page_text)
                pdf_page_text = re.sub(r' ’', '’', pdf_page_text)
                # Remove spaces before a period
                pdf_page_text = re.sub(r'\s+\.', '.', pdf_page_text)
                if pdf_page_text is None or len(pdf_page_text.strip()) == 0:
                    continue
                pages.append(pdf_page_text)
            self._pages = pages

    def __cleanup_pages(self) -> None:
        # Will we store by paragraph or by page?
        pages = []
        for page_num, page in enumerate(self._pages):
            # Remove any remaining newline characters within each page
            page = re.sub(r'\n', ' ', page)
            # Replace multiple consecutive spaces with a single space
            page = re.sub(r'\s+', ' ', page)
            pages.append(page)
        self._pages = pages
        self._flattened_text = [{'text': page, 'chapter': None, 'title': None,
                                 'para_num': None, 'page_num': page_num}
                                for page_num, page in enumerate(self._pages) if len(page.strip()) > 0]

    def __pages_to_paragraphs(self) -> None:
        # Will we store by paragraph or by paragraph?
        paragraphs = []
        for page_num, page in enumerate(self._pages):
            # Get paragraphs on this paragraph
            page_paragraphs = re.split(r'\n(?=[A-Z0-9])|\n\*', page)
            # Remove any remaining newline characters within each paragraph
            page_paragraphs = [re.sub(r'\n', ' ', para) for para in page_paragraphs]
            # Replace multiple consecutive spaces with a single space
            page_paragraphs = [re.sub(r'\s+', ' ', para) for para in page_paragraphs]
            # Create a dict version of the paragraphs with paragraph numbers, etc
            page_paragraph_dicts = [
                {'text': para.strip(), 'para_num': para_num + 1, 'page_num': page_num + 1}
                for para_num, para in enumerate(page_paragraphs)
                if len(para.strip()) > self._min_character_filter
            ]
            if len(page_paragraph_dicts) == 0:
                continue
            paragraphs.extend(page_paragraph_dicts)
        if self._combine_paragraphs:
            self.__format_page_paragraphs(paragraphs)
        # Create flattened text
        self._flattened_text = [{'text': paragraph['text'], 'chapter': None, 'title': None,
                                 'para_num': para_num, 'page_num': paragraph['page_num']}
                                for para_num, paragraph in enumerate(paragraphs) if len(paragraph['text'].strip()) > 0]

    def __create_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [text.replace("\n", " ") for text in texts]
        return self._model.encode(texts)

    def __format_chapter_paragraphs(self) -> None:
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

    def __format_page_paragraphs(self, paragraphs) -> List:
        # Split paragraphs that are too long and merge paragraphs that are too short
        for i, paragraph in enumerate(paragraphs):
            words = paragraph['text'].split()
            if len(words) > self._max_words:
                # Split the paragraph into two
                # Insert paragraph with max words in place of the old paragraph
                maxed_paragraph = ' '.join(words[:self._max_words])
                paragraphs[i] = maxed_paragraph
                # Insert a new paragraph with the remaining words
                new_paragraph = ' '.join(words[self._max_words:])
                paragraphs.insert(i + 1, new_paragraph)

            # Merge paragraphs that are too short
            while len(paragraphs[i]['text'].split()) < self._min_words and i + 1 < len(paragraphs):
                # This paragraph is too short, so merge it with the next one
                paragraphs[i]['text'] += '\n' + paragraphs[i + 1]['text']
                # Delete the next paragraph since we just merged it to the previous one
                del paragraphs[i + 1]

        # After the loop, handle the case where the last paragraph is too short
        last_index = len(paragraphs) - 1
        last_para_len = len(paragraphs[last_index]['text'].split())
        prev_para_len = len(paragraphs[last_index - 1]['text'].split()) if last_index > 0 else 0
        if last_para_len < self._min_words and last_index > 0 and prev_para_len + last_para_len < self._max_words:
            # Merge the last paragraph with the previous one
            paragraphs[last_index - 1]['text'] += '\n ' + paragraphs[last_index]['text']
            # Remove the last paragraph since we just merged it to the previous one
            del paragraphs[last_index]

        # Remove empty paragraphs and whitespace
        paragraphs = [para for para in paragraphs if len(para['text'].strip()) > 0]
        return paragraphs

    def __print_previews(self) -> None:
        for (i, chapter) in enumerate(self._chapters):
            title = chapter['title']
            wc = len(' '.join(chapter['paragraphs']).split(' '))
            paras = len(chapter['paragraphs'])
            initial = chapter['paragraphs'][0][:100]
            preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
            print(preview)

    def __strip_blank_chapters(self) -> None:
        # This takes a list of _chapters and removes the ones outside the range [epub_first_chapter, epub_last_chapter]
        # or if the chapter is too small (likely a title page or something)
        last_chapter = min(self._last_chapter, len(self._chapters) - 1)
        chapters = self._chapters[self._first_chapter:last_chapter + 1]
        # Filter out _chapters with small total paragraph size
        chapters = [chapter for chapter in chapters if sum(len(paragraph) for paragraph
                                                           in chapter['paragraphs']) >= self._min_chapter_size]
        self._chapters = chapters


def test_ebook_search(do_preview=False):
    # noinspection SpellCheckingInspection
    # book_path = \
    #     r'D:\Documents\Books\Karl Popper - The Logic of Scientific Discovery-Routledge (2002)(pdf).pdf'
    book_path = \
        r'D:\Documents\Books\Karl Popper - The Logic of Scientific Discovery-Routledge (2002)(epub).epub'
    # book_path = r"D:\Documents\Books\KJV.epub"
    ebook_search = SemanticSearch(by_paragraph=True, combine_paragraphs=True, page_offset=23)
    if not do_preview:
        ebook_search.load_file(book_path)
        query = 'Why do we need to corroborate theories at all?'
        ebook_search.search(query, top_results=5)
    else:
        ebook_search.preview_epub()


if __name__ == "__main__":
    test_ebook_search(do_preview=False)
    pass
