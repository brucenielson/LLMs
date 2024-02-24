# https://colab.research.google.com/drive/1PDT-jho3Y8TBrktkFVWFAPlc7PaYvlUG?usp=sharing
# https://colab.research.google.com/drive/1PDT-jho3Y8TBrktkFVWFAPlc7PaYvlUG?usp=sharing#scrollTo=zCJx4wZ7fSAB

from sentence_transformers import SentenceTransformer, util
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from os.path import exists
import numpy as np
import math
import os

min_words_per_para = 150
max_words_per_para = 500
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')


def part_to_chapter(part):
    soup = BeautifulSoup(part.get_body_content(), 'html.parser')
    paragraphs = [para.get_text().strip() for para in soup.find_all('p')]
    paragraphs = [para for para in paragraphs if len(para) > 0]
    if len(paragraphs) == 0:
        return None
    title = ' '.join([heading.get_text() for heading in soup.find_all('h1')])
    return {'title': title, 'paras': paragraphs}


def format_paras(chapters):
    for i in range(len(chapters)):
        for j in range(len(chapters[i]['paras'])):
            split_para = chapters[i]['paras'][j].split()
            if len(split_para) > max_words_per_para:
                chapters[i]['paras'].insert(j + 1, ' '.join(split_para[max_words_per_para:]))
                chapters[i]['paras'][j] = ' '.join(split_para[:max_words_per_para])
            k = j
            while len(chapters[i]['paras'][j].split()) < min_words_per_para and k < len(chapters[i]['paras']) - 1:
                chapters[i]['paras'][j] += '\n' + chapters[i]['paras'][k + 1]
                chapters[i]['paras'][k + 1] = ''
                k += 1

        chapters[i]['paras'] = [para.strip() for para in chapters[i]['paras'] if len(para.strip()) > 0]
        if len(chapters[i]['title']) == 0:
            chapters[i]['title'] = '(Unnamed) Chapter {no}'.format(no=i + 1)


def print_previews(chapters):
    for (i, chapter) in enumerate(chapters):
        title = chapter['title']
        wc = len(' '.join(chapter['paras']).split(' '))
        paras = len(chapter['paras'])
        initial = chapter['paras'][0][:30]
        preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
        print(preview)


def get_chapters(book_path, print_chapter_previews, first_chapter, last_chapter):
    book = epub.read_epub(book_path)
    item_doc = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    parts = list(item_doc)
    chapters = [part_to_chapter(part) for part in parts if part_to_chapter(part) is not None]
    last_chapter = min(last_chapter, len(chapters) - 1)
    chapters = chapters[first_chapter:last_chapter + 1]
    format_paras(chapters)
    if print_chapter_previews:
        print_previews(chapters)
    return chapters


def get_embeddings(texts):
    if type(texts) == str:
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    return model.encode(texts)


def read_json(json_path):
    print('Loading embeddings from "{}"'.format(json_path))
    with open(json_path, 'r') as f:
        values = json.load(f)
    return (values['chapters'], np.array(values['embeddings']))


def read_epub(book_path, json_path, preview_mode, first_chapter, last_chapter):
    chapters = get_chapters(book_path, preview_mode, first_chapter, last_chapter)
    if preview_mode:
        return (chapters, None)
    print('Generating embeddings for chapters {}-{} in "{}"\n'.format(first_chapter, last_chapter, book_path))
    paras = [para for chapter in chapters for para in chapter['paras']]
    embeddings = get_embeddings(paras)
    try:
        with open(json_path, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except:
        print('Failed to save embeddings to "{}"'.format(json_path))
    return (chapters, embeddings)


def process_file(path, preview_mode=False, first_chapter=0, last_chapter=math.inf):
    values = None
    if path[-4:] == 'json':
        values = read_json(path)
    elif path[-4:] == 'epub':
        json_path = 'embeddings-{}-{}-{}.json'.format(first_chapter, last_chapter, path)
        if exists(json_path):
            values = read_json(json_path)
        else:
            values = read_epub(path, json_path, preview_mode, first_chapter, last_chapter)
    else:
        print('Invalid file format. Either upload an epub or a json of book embeddings.')
    return values


def print_and_write(text, f):
    print(text)
    f.write(text + '\n')


def index_to_para_chapter_index(index, chapters):
    for chapter in chapters:
        paras_len = len(chapter['paras'])
        if index < paras_len:
            return chapter['paras'][index], chapter['title'], index
        index -= paras_len
    return None


def search(query, embeddings, path, chapters, n=3):
    query_embedding = get_embeddings(query)[0]
    scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    results = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]

    f = open('result.text', 'a')
    header_msg ='Results for query "{}" in "{}"'.format(query, path)
    print_and_write(header_msg, f)
    for index in results:
        para, title, para_no = index_to_para_chapter_index(index, chapters)
        result_msg = '\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'.format(title, para_no, scores[index], para)
        print_and_write(result_msg, f)
    print_and_write('\n', f)


def embed_epub(epub_file_name, preview_mode=False, first_chapter=0, last_chapter=math.inf):
    assert get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
    json_file_name = switch_ext(epub_file_name, '.json')
    chapters = get_chapters(epub_file_name, preview_mode, first_chapter, last_chapter)
    if preview_mode:
        return chapters, None
    print('Generating embeddings for chapters {}-{} in "{}"\n'
          .format(first_chapter, last_chapter, epub_file_name))
    paras = [para for chapter in chapters for para in chapter['paras']]
    embeddings = get_embeddings(paras)
    try:
        with open(json_file_name, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except IOError as io_error:
        print(f'Failed to save embeddings to "{json_file_name}": {io_error}')
    except json.JSONDecodeError as json_error:
        print(f'Failed to decode JSON in "{json_file_name}": {json_error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    return chapters, embeddings


def create_embeddings_json(epub_file_name):
    # Assert this is an epub file
    assert get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'

    json_file_name = switch_ext(epub_file_name, '.json')

    if exists(json_file_name):
        # Delete it if it exists
        os.remove(json_file_name)

    chapters, embeddings = embed_epub(epub_file_name, preview_mode=False)
    return embeddings


def get_ext(full_file_name):
    return full_file_name[full_file_name.rfind('.'):]


def switch_ext(full_file_name, new_ext):
    return full_file_name[:full_file_name.rfind('.')] + new_ext


def ebook_semantic_search(query, file):

    # First check if we have an embeddings file for this epub which is a json file with the same name as the epub
    if get_ext(file) == '.epub':
        json_file = switch_ext(file, '.json')
        if not exists(json_file):
            create_embeddings_json(file)
        file = json_file

    assert get_ext(file) == '.json', 'Should now be a json file.'
    # Load the embeddings from the json file
    chapters, embeddings = process_file(file, preview_mode=False)
    search(query, embeddings, file, chapters)


if __name__ == "__main__":
    book_path = \
        r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).epub'
    ebook_semantic_search('Why do we need to corroborate theories at all?', book_path)
