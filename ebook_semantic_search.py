from sentence_transformers import SentenceTransformer
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from os.path import exists
import numpy as np
import math
import os


class SemanticSearch:

    # noinspection SpellCheckingInspection
    def __init__(self, full_file_name=None, model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1',
                 do_strip=True, min_chapter_size=2000, first_chapter=0, last_chapter=math.inf,
                 min_words_per_paragraph=150, max_words_per_paragraph=500):
        self._model = SentenceTransformer(model_name)
        self._file_name = full_file_name
        self._do_strip = do_strip
        self._min_chapter_size = min_chapter_size
        self._first_chapter = first_chapter
        self._last_chapter = last_chapter
        self._min_words = min_words_per_paragraph
        self._max_words = max_words_per_paragraph
        self._chapters = None
        self._embeddings = None

        if full_file_name is not None:
            self.load_file(full_file_name)

    @staticmethod
    def get_ext(full_file_name):
        return full_file_name[full_file_name.rfind('.'):]

    @staticmethod
    def switch_ext(full_file_name, new_ext):
        return full_file_name[:full_file_name.rfind('.')] + new_ext

    @staticmethod
    def read_json(json_path):
        print('Loading _embeddings from "{}"'.format(json_path))
        with open(json_path, 'r') as f:
            values = json.load(f)
        return values['_chapters'], np.array(values['_embeddings'])

    @staticmethod
    def print_and_write(text, f):
        print(text)
        f.write(text + '\n')

    @staticmethod
    def score(query_embedding, embeddings):
        # Compute the cosine similarity between the query and all paragraphs
        scores = (np.dot(embeddings, query_embedding) /
                  (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)))
        return scores

    @staticmethod
    def epub_sections_to_chapter(section):
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
    def do_strip(self):
        return self._do_strip

    @do_strip.setter
    def do_strip(self, value):
        self._do_strip = value

    @property
    def min_chapter_size(self):
        return self._min_chapter_size

    @min_chapter_size.setter
    def min_chapter_size(self, value):
        self._min_chapter_size = value

    @property
    def first_chapter(self):
        return self._first_chapter

    @first_chapter.setter
    def first_chapter(self, value):
        self._first_chapter = value

    @property
    def last_chapter(self):
        return self._last_chapter

    @last_chapter.setter
    def last_chapter(self, value):
        self._last_chapter = value

    @property
    def min_words_per_paragraph(self):
        return self._min_words

    @min_words_per_paragraph.setter
    def min_words_per_paragraph(self, value):
        self._min_words = value

    @property
    def max_words_per_paragraph(self):
        return self._max_words

    @max_words_per_paragraph.setter
    def max_words_per_paragraph(self, value):
        self._max_words = value

    def load_model(self, model_name):
        self._model = SentenceTransformer(model_name)

    def load_file(self, full_file_name):
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
                self.embed_epub(self._file_name)
            else:
                self._file_name = json_file_name
        # A json file should now exist with our embeddings
        assert self.get_ext(self._file_name) == '.json', 'Should now be a json file.'
        # Do we now have embeddings and chapters? If not, load them
        if self._chapters is None or self._embeddings is None:
            self.load_embeddings_file()

    def format_paragraphs(self):
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
                while len(chapter['paragraphs'][j].split()) < self._min_words and j+1 < len(chapter['paragraphs']):
                    # This paragraph is too short, so merge it with the next one
                    chapter['paragraphs'][j] += '\n' + chapter['paragraphs'][j+1]
                    # Delete the next paragraph since we just merged it to the previous one
                    del chapter['paragraphs'][j+1]

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

    def print_previews(self):
        for (i, chapter) in enumerate(self._chapters):
            title = chapter['title']
            wc = len(' '.join(chapter['paragraphs']).split(' '))
            paras = len(chapter['paragraphs'])
            initial = chapter['paragraphs'][0][:100]
            preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
            print(preview)

    @staticmethod
    def index_to_para_chapter_index(index, chapters):
        for chapter in chapters:
            paras_len = len(chapter['paragraphs'])
            if index < paras_len:
                return chapter['paragraphs'][index], chapter['title'], index
            index -= paras_len
        return None

    def strip_blank_chapters(self):
        # This takes a list of _chapters and removes the ones outside the range [first_chapter, last_chapter]
        # or if the chapter is too small (likely a title page or something)
        last_chapter = min(self._last_chapter, len(self._chapters) - 1)
        chapters = self._chapters[self._first_chapter:last_chapter + 1]
        # Filter out _chapters with small total paragraph size
        chapters = [chapter for chapter in chapters if sum(len(paragraph) for paragraph
                                                           in chapter['paragraphs']) >= self._min_chapter_size]
        self._chapters = chapters

    def create_embeddings(self, texts):
        if type(texts) is str:
            texts = [texts]
        texts = [text.replace("\n", " ") for text in texts]
        return self._model.encode(texts)

    def load_json_file(self, path):
        assert self.get_ext(path) == '.json', 'Invalid file format. Please upload a json file.'
        return self.read_json(path)

    def search(self, query, top_results=5):
        # Create _embeddings for the query
        query_embedding = self.create_embeddings(query)[0]
        scores = self.score(query_embedding, self._embeddings)
        results = sorted([i for i in range(len(self._embeddings))], key=lambda i: scores[i], reverse=True)[:top_results]
        # Write out the results
        f = open('result.text', 'a')
        header_msg = 'Results for query "{}" in "{}"'.format(query, self._file_name)
        self.print_and_write(header_msg, f)
        for index in results:
            para, title, para_no = self.index_to_para_chapter_index(index, self._chapters)
            result_msg = ('\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'
                          .format(title, para_no, scores[index], para))
            self.print_and_write(result_msg, f)
        self.print_and_write('\n', f)

    def epub_to_chapters(self, epub_file_name):
        book = epub.read_epub(epub_file_name)
        item_doc = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        book_sections = list(item_doc)
        chapters = [result for section in book_sections
                    if (result := self.epub_sections_to_chapter(section)) is not None]
        self._chapters = chapters
        if self._do_strip:
            self.strip_blank_chapters()
        self.format_paragraphs()

    def embed_epub(self, epub_file_name):
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
        self.epub_to_chapters(epub_file_name)
        print('Generating _embeddings for "{}"\n'.format(epub_file_name))
        paragraphs = [paragraph for chapter in self._chapters for paragraph in chapter['paragraphs']]
        # Generate the _embeddings using the _model
        self._embeddings = self.create_embeddings(paragraphs)
        # Save the chapter _embeddings to a json file
        try:
            print('Writing _embeddings for "{}"\n'.format(epub_file_name))
            with open(json_file_name, 'w') as f:
                json.dump({'_chapters': self._chapters, '_embeddings': self._embeddings.tolist()}, f)
            self._file_name = json_file_name
        except IOError as io_error:
            print(f'Failed to save _embeddings to "{json_file_name}": {io_error}')
        except json.JSONDecodeError as json_error:
            print(f'Failed to decode JSON in "{json_file_name}": {json_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

    def preview_epub(self):
        epub_file_name = self._file_name
        assert self.get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
        self.epub_to_chapters(epub_file_name)
        self.print_previews()

    def load_embeddings_file(self):
        assert self.get_ext(self._file_name) == '.json', 'Should now be a json file.'
        # Load the _embeddings from the json file
        self._chapters, self._embeddings = self.load_json_file(self._file_name)


def test_ebook_search(do_preview=False):
    # noinspection SpellCheckingInspection
    book_path = \
        r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).epub'
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
