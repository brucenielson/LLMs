from sentence_transformers import SentenceTransformer
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from os.path import exists
import numpy as np
import math
import os


class EBookSearch:

    # noinspection SpellCheckingInspection
    def __init__(self, full_file_name, model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        self.model = SentenceTransformer(model_name)
        self.full_file_name = full_file_name
        self.chapters = None
        self.embeddings = None

    @staticmethod
    def format_paragraphs(chapters, min_words_per_para=150, max_words_per_para=500):
        for i in range(len(chapters)):
            for j in range(len(chapters[i]['paragraphs'])):
                split_para = chapters[i]['paragraphs'][j].split()
                if len(split_para) > max_words_per_para:
                    chapters[i]['paragraphs'].insert(j + 1, ' '.join(split_para[max_words_per_para:]))
                    chapters[i]['paragraphs'][j] = ' '.join(split_para[:max_words_per_para])
                k = j
                while (len(chapters[i]['paragraphs'][j].split()) < min_words_per_para
                       and k < len(chapters[i]['paragraphs']) - 1):
                    chapters[i]['paragraphs'][j] += '\n' + chapters[i]['paragraphs'][k + 1]
                    chapters[i]['paragraphs'][k + 1] = ''
                    k += 1

            chapters[i]['paragraphs'] = [para.strip() for para in chapters[i]['paragraphs'] if len(para.strip()) > 0]
            if len(chapters[i]['title']) == 0:
                chapters[i]['title'] = '(Unnamed) Chapter {no}'.format(no=i + 1)

    @staticmethod
    def print_previews(chapters):
        for (i, chapter) in enumerate(chapters):
            title = chapter['title']
            wc = len(' '.join(chapter['paragraphs']).split(' '))
            paras = len(chapter['paragraphs'])
            initial = chapter['paragraphs'][0][:30]
            preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
            print(preview)

    @staticmethod
    def get_ext(full_file_name):
        return full_file_name[full_file_name.rfind('.'):]

    @staticmethod
    def switch_ext(full_file_name, new_ext):
        return full_file_name[:full_file_name.rfind('.')] + new_ext

    @staticmethod
    def read_json(json_path):
        print('Loading embeddings from "{}"'.format(json_path))
        with open(json_path, 'r') as f:
            values = json.load(f)
        return values['chapters'], np.array(values['embeddings'])

    @staticmethod
    def print_and_write(text, f):
        print(text)
        f.write(text + '\n')

    @staticmethod
    def index_to_para_chapter_index(index, chapters):
        for chapter in chapters:
            paras_len = len(chapter['paragraphs'])
            if index < paras_len:
                return chapter['paragraphs'][index], chapter['title'], index
            index -= paras_len
        return None

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
        text_list = [para.get_text().strip() for para in p_tag_list if para.get_text().strip()]
        if len(text_list) == 0:
            return None
        # Extract and process headings
        heading_list = [heading.get_text().strip() for heading in html.find_all('h1')]
        title = ' '.join(heading_list)
        return {'title': title, 'paragraphs': text_list}

    @staticmethod
    def strip_blank_chapters(chapters, first_chapter=0, last_chapter=math.inf, min_paragraph_size=2000):
        # This takes a list of chapters and removes the ones outside the range [first_chapter, last_chapter]
        last_chapter = min(last_chapter, len(chapters) - 1)
        chapters = chapters[first_chapter:last_chapter + 1]
        # Filter out chapters with small total paragraph size
        chapters = [chapter for chapter in chapters if sum(len(para) for para
                                                           in chapter['paragraphs']) >= min_paragraph_size]
        return chapters

    def create_embeddings(self, texts):
        if type(texts) is str:
            texts = [texts]
        texts = [text.replace("\n", " ") for text in texts]
        return self.model.encode(texts)

    def load_json_file(self, path):
        assert self.get_ext(path) == '.json', 'Invalid file format. Please upload a json file.'
        return self.read_json(path)

    def search(self, query, embeddings, path, chapters, n=5):
        # Create embeddings for the query
        query_embedding = self.create_embeddings(query)[0]
        scores = self.score(query_embedding, embeddings)
        results = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]
        # Write out the results
        f = open('result.text', 'a')
        header_msg = 'Results for query "{}" in "{}"'.format(query, path)
        self.print_and_write(header_msg, f)
        for index in results:
            para, title, para_no = self.index_to_para_chapter_index(index, chapters)
            result_msg = ('\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'
                          .format(title, para_no, scores[index], para))
            self.print_and_write(result_msg, f)
        self.print_and_write('\n', f)

    def epub_to_chapters(self, epub_file_name, do_strip=True, first_chapter=0, last_chapter=math.inf):
        book = epub.read_epub(epub_file_name)
        item_doc = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        book_sections = list(item_doc)
        chapters = [result for section in book_sections
                    if (result := self.epub_sections_to_chapter(section)) is not None]
        if do_strip:
            chapters = self.strip_blank_chapters(chapters, first_chapter, last_chapter)
        self.format_paragraphs(chapters)
        return chapters

    def embed_epub(self, epub_file_name, do_strip=True, first_chapter=0, last_chapter=math.inf):
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
        chapters = self.epub_to_chapters(epub_file_name, do_strip, first_chapter, last_chapter)
        print('Generating embeddings for "{}"\n'.format(epub_file_name))
        paragraphs = [paragraph for chapter in chapters for paragraph in chapter['paragraphs']]
        # Generate the embeddings using the model
        embeddings = self.create_embeddings(paragraphs)
        # Save the chapter embeddings to a json file
        try:
            print('Writing embeddings for "{}"\n'.format(epub_file_name))
            with open(json_file_name, 'w') as f:
                json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
        except IOError as io_error:
            print(f'Failed to save embeddings to "{json_file_name}": {io_error}')
        except json.JSONDecodeError as json_error:
            print(f'Failed to decode JSON in "{json_file_name}": {json_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
        # Return the chapters and the embeddings
        return chapters, embeddings

    def preview_epub(self, do_strip=True):
        epub_file_name = self.full_file_name
        assert self.get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
        chapters = self.epub_to_chapters(epub_file_name, do_strip)
        self.print_previews(chapters)
        return chapters

    def ebook_preview(self, do_strip=True):
        self.preview_epub(do_strip)

    def load_embeddings(self, do_strip=True, first_chapter=0, last_chapter=math.inf):
        file = self.full_file_name
        # First check if we have an embeddings file for this epub which is a json file with the same name as the epub
        if self.get_ext(file) == '.epub':
            json_file = self.switch_ext(file, '.json')
            if not exists(json_file):
                self.embed_epub(file, do_strip, first_chapter, last_chapter)
            file = json_file

        assert self.get_ext(file) == '.json', 'Should now be a json file.'
        # Load the embeddings from the json file
        chapters, embeddings = self.load_json_file(file)
        self.chapters = chapters
        self.embeddings = embeddings

    def ebook_semantic_search(self, query, do_strip=True, first_chapter=0, last_chapter=math.inf):
        self.load_embeddings(do_strip, first_chapter, last_chapter)
        if self.embeddings is not None:
            self.search(query, self.embeddings, self.full_file_name, self.chapters)


def test_ebook_search(do_preview=False, do_strip=True):
    # noinspection SpellCheckingInspection
    book_path = \
        r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).epub'
    ebook_search = EBookSearch(book_path)
    if do_preview:
        ebook_search.ebook_preview(do_strip=do_strip)
        return
    else:
        query = 'Why do we need to corroborate theories at all?'
        ebook_search.ebook_semantic_search(query, do_strip=do_strip)


if __name__ == "__main__":
    test_ebook_search(do_preview=False)
