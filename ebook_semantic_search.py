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
import logging
from typing import List, Optional, Tuple, Union
from typing import TextIO
from pypdf import PdfReader
import re


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
        self._flattened_paragraphs = None
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
        assert self.get_ext(full_file_name) in ['.epub', '.json', '.pdf'], ('Invalid file format. '
                                                                            'Please upload an epub or json file.')
        # Reset all the variables on load
        self._chapters = None
        self._embeddings = None
        self._flattened_paragraphs = None
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
                paragraph, title, paragraph_num, page_num = self.__index_into_chapters(i)
                if page_num is None:
                    result_msg = ('\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'
                                  .format(title, paragraph_num, scores[i], paragraph))
                else:
                    result_msg = ('\nPage number: {}, Passage number: {}, Score: {:.2f}\n"{}"'
                                  .format(page_num, paragraph_num, scores[i], paragraph))
                results_msgs.append(result_msg)
                self.print_and_write(result_msg, f)
            self.print_and_write('\n', f)
        return results_msgs, results

    def preview_epub(self) -> None:
        epub_file_name = self._file_name
        assert self.get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
        self.__file_to_chapters(epub_file_name)
        self.__print_previews()

    def __embed_file(self, epub_file_name: str) -> None:
        # Assert this is an epub file
        assert self.get_ext(epub_file_name) in ['.epub', '.pdf'], 'Invalid file format. Please upload an epub file.'
        # Create a json file with the same name as the epub
        json_file_name = self.switch_ext(epub_file_name, '.json')
        # Delete any existing json file with the same name
        if exists(json_file_name):
            os.remove(json_file_name)
        # Create the json file name
        json_file_name = self.switch_ext(epub_file_name, '.json')
        # Convert the epub to html and extract the paragraphs
        self.__file_to_chapters(epub_file_name)
        print('Generating embeddings for "{}"\n'.format(epub_file_name))
        # Handle epub (chapters) vs pdf (pages only)
        paragraphs = []
        text_list = []
        if self._flattened_paragraphs is None and self._chapters is not None:
            paragraphs = [paragraph for chapter in self._chapters for paragraph in chapter['paragraphs']]
            text_list = self._chapters
        elif self._flattened_paragraphs is not None:
            paragraphs = [para['text'] for para in self._flattened_paragraphs]
            text_list = self._flattened_paragraphs
        # Generate the _embeddings using the _model
        self._embeddings = self.__create_embeddings(paragraphs)
        # Save the chapter _embeddings to a json file
        try:
            print('Writing embeddings for "{}"\n'.format(epub_file_name))
            with open(json_file_name, 'w') as f:
                json.dump({'_chapters': text_list, '_embeddings': self._embeddings.tolist()}, f)
            self._file_name = json_file_name
        except IOError as io_error:
            print(f'Failed to save embeddings to "{json_file_name}": {io_error}')
        except json.JSONDecodeError as json_error:
            print(f'Failed to decode JSON in "{json_file_name}": {json_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

    def __index_into_chapters(self, index: int) -> Tuple[str, Optional[str], int, Optional[int]]:
        if self._flattened_paragraphs is None:
            self._flattened_paragraphs = [{'text': paragraph, 'title': chapter['title'], 'para_num': para_num,
                                           'page_num': None}
                                          for chapter in self._chapters
                                          for para_num, paragraph in enumerate(chapter['paragraphs'])]

        return (self._flattened_paragraphs[index]['text'], self._flattened_paragraphs[index]['title'],
                self._flattened_paragraphs[index]['para_num'], self._flattened_paragraphs[index]['page_num']
                if 0 <= index < len(self._flattened_paragraphs) else None)

    def __file_to_chapters(self, file_path: str) -> None:
        file_ext = self.get_ext(file_path)
        if file_ext == '.epub':
            self.__epub_to_chapters(file_path)
        elif file_ext == '.pdf':
            self.__pdf_to_pages(file_path)
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
        self.__format_paragraphs()

    def __pdf_to_pages(self, pdf_file_path: str) -> None:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            flattened_paragraphs = []
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
                # Get paragraphs on this page
                page_paragraphs = re.split(r'\n(?=[A-Z0-9])|\n\*', pdf_page_text)
                # Remove any remaining newline characters within each paragraph
                page_paragraphs = [re.sub(r'\n', ' ', para) for para in page_paragraphs]
                # Replace multiple consecutive spaces with a single space
                page_paragraphs = [re.sub(r'\s+', ' ', para) for para in page_paragraphs]
                # Create a dict version of the paragraphs with page numbers, etc
                page_paragraph_dicts = [
                    {'text': para.strip(), 'chapter': None, 'title': None, 'para_num': para_num + 1,
                     'page_num': page_num + 1}
                    for para_num, para in enumerate(page_paragraphs)
                    if len(para.strip()) > 25
                ]
                if len(page_paragraph_dicts) == 0:
                    continue
                flattened_paragraphs.extend(page_paragraph_dicts)

        self._flattened_paragraphs = flattened_paragraphs

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

    def __format_flattened_paragraphs(self, paragraphs: list) -> None:
        # Split paragraphs that are too long and merge paragraphs that are too short
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            words = paragraph.split()

            if len(words) > self._max_words:
                # Split the paragraph into two
                # Insert paragraph with max words in place of the old paragraph
                maxed_paragraph = ' '.join(words[:self._max_words])
                paragraphs[i] = maxed_paragraph

                # Insert a new paragraph with the remaining words
                new_paragraph = ' '.join(words[self._max_words:])
                paragraphs.insert(i + 1, new_paragraph)

            # Merge paragraphs that are too short
            while len(paragraphs[i].split()) < self._min_words and i + 1 < len(
                    paragraphs):
                # This paragraph is too short, so merge it with the next one
                paragraphs[i] += ' ' + paragraphs[i + 1]
                # Delete the next paragraph since we just merged it to the previous one
                del paragraphs[i + 1]

            i += 1

        # After the loop, handle the case where the last paragraph is too short
        last_index = len(paragraphs) - 1
        last_para_len = len(paragraphs[last_index].split())
        prev_para_len = len(paragraphs[last_index - 1].split()) if last_index > 0 else 0

        if last_para_len < self._min_words and last_index > 0 and prev_para_len + last_para_len < self._max_words:
            # Merge the last paragraph with the previous one
            paragraphs[last_index - 1] += ' ' + paragraphs[last_index]
            # Remove the last paragraph since we just merged it to the previous one
            del paragraphs[last_index]

        # # Remove empty paragraphs and whitespace
        # self._flattened_paragraphs = [
        #     {'text': para['text'].strip(), 'title': '', 'para_num': para['para_num']}
        #     for para in self._flattened_paragraphs
        #     if len(para['text'].strip()) > 0
        # ]

        self._flattened_paragraphs = [{'text': paragraph, 'title': chapter['title'], 'para_num': para_num}
                                      for chapter in self._chapters
                                      for para_num, paragraph in enumerate(chapter['paragraphs'])]

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


def replace_ligatures(text):
    # noinspection SpellCheckingInspection
    ligature_mapping = {
        '\ufb00': 'ff',
        '\ufb01': 'fi',
        '\ufb02': 'fl',
        '\ufb03': 'ffi',
        '\ufb04': 'ffl',
        '\ufb05': 'st',
        '\ufb06': 'st',
        '\ufb07': 'ct',
        '\ufb08': 'st',
        '\ufb09': 'st',
        '\ufb0a': 'et',
        '\ufb13': 'ij',
        '\ufb15': 'ij',
        '\ufb1d': 'oe',
        '\ufb1e': 'oe',
        '\ufb1f': 'oe',
        '\ufb20': 'b',
        '\ufb21': 's',
        '\ufb22': 'B',
        '\ufb23': 'P',
        '\ufb24': 'o',
        '\ufb25': 'C',
        '\ufb26': 'c',
        '\ufb27': 'd',
        '\ufb28': 'D',
        '\ufb29': 'e',
        '\ufb2a': 'e',
        '\ufb2b': 'e',
        '\ufb2c': 'e',
        '\ufb2d': 'j',
        '\ufb2e': 'g',
        '\ufb2f': 'G',
        '\ufb30': 'h',
        '\ufb31': 'H',
        '\ufb32': 'i',
        '\ufb33': 'I',
        '\ufb34': 'I',
        '\ufb35': 'l',
        '\ufb36': 'L',
        '\ufb38': 'N',
        '\ufb39': 'n',
        '\ufb3a': 'O',
        '\ufb3b': 'O',
        '\ufb3c': 'O',
        '\ufb3e': 'r',
        '\ufb3f': 'r',
        '\ufb40': 'R',
        '\ufb41': 'R',
        '\ufb42': 'R',
        '\ufb43': 'S',
        '\ufb44': 's',
        '\ufb45': 'S',
        '\ufb46': 's',
        '\ufb47': 't',
        '\ufb48': 'T',
        '\ufb49': 'U',
        '\ufb4a': 'u',
        '\ufb4b': 'V',
        '\ufb4c': 'Y',
        '\ufb4d': 'y',
        '\ufb4e': 'W',
        '\ufb4f': 'w',
        '\ufb50': 'A',
        '\ufb51': 'a',
        '\ufb52': 'B',
        '\ufb53': 'b',
        '\ufb54': 'B',
        '\ufb56': 'e',
        '\ufb57': 'e',
        '\ufb58': 'F',
        '\ufb59': 'f',
        '\ufb5a': 'G',
        '\ufb5b': 'g',
        '\ufb5c': 'H',
        '\ufb5d': 'h',
        '\ufb5e': 'I',
        '\ufb5f': 'i',
        '\ufb60': 'I',
        '\ufb62': 'i',
        '\ufb63': 'j',
        '\ufb64': 'k',
        '\ufb65': 'k',
        '\ufb66': 'l',
        '\ufb67': 'l',
        '\ufb68': 'l',
        '\ufb69': 'l',
        '\ufb6a': 'N',
        '\ufb6b': 'n',
        '\ufb6c': 'O',
        '\ufb6d': 'O',
        '\ufb6e': 'o',
        '\ufb6f': 'o',
        '\ufb70': 'o',
        '\ufb71': 'o',
        '\ufb72': 'o',
        '\ufb73': 'o',
        '\ufb74': 'P',
        '\ufb75': 'p',
        '\ufb76': 'P',
        '\ufb77': 'p',
        '\ufb78': 'R',
        '\ufb79': 'r',
        '\ufb7a': 'r',
        '\ufb7b': 'r',
        '\ufb7c': 'r',
        '\ufb7d': 'r',
        '\ufb7e': 'S',
        '\ufb7f': 's',
        '\ufb80': 'S',
        '\ufb81': 's',
        '\ufb82': 'S',
        '\ufb83': 's',
        '\ufb84': 'S',
        '\ufb85': 's',
        '\ufb86': 't',
        '\ufb87': 't',
        '\ufb88': 'T',
        '\ufb89': 't',
        '\ufb8a': 'T',
        '\ufb8b': 'U',
        '\ufb8c': 'u',
        '\ufb8d': 'U',
        '\ufb8e': 'u',
        '\ufb8f': 'u',
        '\ufb90': 'v',
        '\ufb91': 'v',
        '\ufb92': 'w',
        '\ufb93': 'w',
        '\ufb94': 'Y',
        '\ufb95': 'y',
        '\ufb96': 'Y',
        '\ufb97': 'A',
        '\ufb98': 'a',
        '\ufb99': 'B',
        '\ufb9a': 'b',
        '\ufb9b': 'O',
        '\ufb9c': 'o',
        '\ufb9d': 'O',
        '\ufb9e': 'o',
        '\ufb9f': 'O',
        '\ufba0': 'o',
        '\ufba1': 'o',
        '\ufba2': 'o',
        '\ufba3': 'o',
        '\ufba4': 'p',
        '\ufba5': 't',
        '\ufba6': 'P',
        '\ufba7': 'p',
        '\ufba8': 'p',
        '\ufba9': 'r',
        '\ufbaa': 'R',
        '\ufbab': 'r',
        '\ufbac': 'r',
        '\ufbad': 'R',
        '\ufbae': 'R',
        '\ufbaf': 'r',
        '\ufbb0': 'S',
        '\ufbb1': 's',
        '\ufbb2': 'S',
        '\ufbb3': 's',
        '\ufbb4': 'S',
        '\ufbb5': 's',
        '\ufbb6': 'T',
        '\ufbb7': 't',
        '\ufbb8': 'T',
        '\ufbb9': 't',
        '\ufbba': 'T',
        '\ufbbb': 't',
        '\ufbbc': 'T',
        '\ufbbd': 'U',
        '\ufbbe': 'u',
        '\ufbbf': 'U',
        '\ufbc0': 'u',
        '\ufbc1': 'U',
        '\ufbc2': 'u',
        '\ufbc3': 'V',
        '\ufbc4': 'v',
        '\ufbc5': 'V',
        '\ufbc6': 'v',
        '\ufbc7': 'W',
        '\ufbc8': 'w',
        '\ufbc9': 'W',
        '\ufbca': 'w',
        '\ufbcb': 'W',
        '\ufbcc': 'w',
        '\ufbcd': 'W',
        '\ufbce': 'w',
        '\ufbcf': 'Y',
        '\ufbd0': 'y',
        '\ufbd1': 'Y',
        '\ufbd2': 'y',
        '\ufbd3': 'Z',
        '\ufbd4': 'z',
        '\ufbd5': 'Z',
        '\ufbd6': 'z',
        '\ufbd7': 'Z',
        '\ufbd8': 'z',
        '\ufbd9': 'Z',
        '\ufbda': 'z',
        '\ufbdb': 'Z',
        '\ufbdc': 'z',
    }

    for ligature, replacement in ligature_mapping.items():
        text = text.replace(ligature, replacement)

    return text


def test_ebook_search(do_preview=False):
    # noinspection SpellCheckingInspection
    book_path = \
        r'D:\Documents\Papers\EPub Books\Karl Popper - The Logic of Scientific Discovery-Routledge (2002).pdf'
    # book_path = r"D:\Documents\Papers\EPub Books\KJV.epub"
    ebook_search = SemanticSearch()
    if not do_preview:
        ebook_search.load_file(book_path)
        query = 'Why do we need to corroborate theories at all?'
        ebook_search.search(query, top_results=5)
    else:
        ebook_search.preview_epub()


if __name__ == "__main__":
    test_ebook_search(do_preview=False)
    pass
