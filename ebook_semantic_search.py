# https://colab.research.google.com/drive/1PDT-jho3Y8TBrktkFVWFAPlc7PaYvlUG?usp=sharing
# https://colab.research.google.com/drive/1PDT-jho3Y8TBrktkFVWFAPlc7PaYvlUG?usp=sharing#scrollTo=zCJx4wZ7fSAB

from sentence_transformers import SentenceTransformer
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


def format_paras(chapters):
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


def print_previews(chapters):
    for (i, chapter) in enumerate(chapters):
        title = chapter['title']
        wc = len(' '.join(chapter['paragraphs']).split(' '))
        paras = len(chapter['paragraphs'])
        initial = chapter['paragraphs'][0][:30]
        preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
        print(preview)


def get_embeddings(texts):
    if type(texts) is str:
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    return model.encode(texts)


def read_json(json_path):
    print('Loading embeddings from "{}"'.format(json_path))
    with open(json_path, 'r') as f:
        values = json.load(f)
    return values['chapters'], np.array(values['embeddings'])


def read_epub(epub_path, json_path, first_chapter, last_chapter):
    try:
        chapters = epub_to_html(epub_path, first_chapter, last_chapter)
        print('Generating embeddings for "{}"\n'.format(epub_path))
        paras = [para for chapter in chapters for para in chapter['paragraphs']]
        embeddings = get_embeddings(paras)
        with open(json_path, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except FileNotFoundError as file_not_found_error:
        print(f"Error: {file_not_found_error}. Please check the file paths.")
    except json.JSONDecodeError as json_error:
        print(f"Error decoding JSON: {json_error}")
    except IOError as io_error:
        print(f"Error during file operations: {io_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # You can also print the exception type to help with debugging: print(type(e))
    else:
        return chapters, embeddings


def process_file(path, first_chapter=0, last_chapter=math.inf):
    values = None
    if path[-4:] == 'json':
        values = read_json(path)
    elif path[-4:] == 'epub':
        json_path = 'embeddings-{}-{}-{}.json'.format(first_chapter, last_chapter, path)
        if exists(json_path):
            values = read_json(json_path)
        else:
            values = read_epub(path, json_path, first_chapter, last_chapter)
    else:
        print('Invalid file format. Either upload an epub or a json of book embeddings.')
    return values


def print_and_write(text, f):
    print(text)
    f.write(text + '\n')


def index_to_para_chapter_index(index, chapters):
    for chapter in chapters:
        paras_len = len(chapter['paragraphs'])
        if index < paras_len:
            return chapter['paragraphs'][index], chapter['title'], index
        index -= paras_len
    return None


def search(query, embeddings, path, chapters, n=3):
    query_embedding = get_embeddings(query)[0]
    scores = (np.dot(embeddings, query_embedding) /
              (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)))
    results = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]

    f = open('result.text', 'a')
    header_msg = 'Results for query "{}" in "{}"'.format(query, path)
    print_and_write(header_msg, f)
    for index in results:
        para, title, para_no = index_to_para_chapter_index(index, chapters)
        result_msg = ('\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'
                      .format(title, para_no, scores[index], para))
        print_and_write(result_msg, f)
    print_and_write('\n', f)


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


def epub_to_html(epub_file_name, first_chapter=0, last_chapter=math.inf):
    book = epub.read_epub(epub_file_name)
    item_doc = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    book_sections = list(item_doc)
    chapters = [result for section in book_sections if (result := epub_sections_to_chapter(section)) is not None]
    chapters = strip_blank_chapters(chapters, first_chapter, last_chapter)
    format_paras(chapters)
    return chapters


def strip_blank_chapters(chapters, first_chapter=0, last_chapter=math.inf, min_paragraph_size=2000):
    # This takes a list of chapters and removes the ones outside the range [first_chapter, last_chapter]
    # It also removes 'empty' chapters, i.e. those that are very small and likely a title page, etc.
    last_chapter = min(last_chapter, len(chapters) - 1)
    chapters = chapters[first_chapter:last_chapter + 1]
    # Filter out chapters with small total paragraph size
    chapters = [chapter for chapter in chapters if sum(len(para) for para
                                                       in chapter['paragraphs']) >= min_paragraph_size]
    return chapters


def embed_epub(epub_file_name, first_chapter=0, last_chapter=math.inf):
    assert get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
    json_file_name = switch_ext(epub_file_name, '.json')
    chapters = epub_to_html(epub_file_name, first_chapter, last_chapter)
    print('Generating embeddings for "{}"\n'
          .format(epub_file_name))
    paras = [para for chapter in chapters for para in chapter['paragraphs']]
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


def preview_epub(epub_file_name, do_strip=True):
    assert get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'
    chapters = epub_to_html(epub_file_name, do_strip)
    print_previews(chapters)
    return chapters


def create_embeddings_json(epub_file_name):
    # Assert this is an epub file
    assert get_ext(epub_file_name) == '.epub', 'Invalid file format. Please upload an epub file.'

    json_file_name = switch_ext(epub_file_name, '.json')

    if exists(json_file_name):
        # Delete it if it exists
        os.remove(json_file_name)

    chapters, embeddings = embed_epub(epub_file_name)
    return embeddings


def get_ext(full_file_name):
    return full_file_name[full_file_name.rfind('.'):]


def switch_ext(full_file_name, new_ext):
    return full_file_name[:full_file_name.rfind('.')] + new_ext


def ebook_semantic_search(query, file, do_preview=False, do_strip=True):
    # First check if we have an embeddings file for this epub which is a json file with the same name as the epub
    if get_ext(file) == '.epub':
        if do_preview:
            preview_epub(file, do_strip)
            return
        json_file = switch_ext(file, '.json')
        if not exists(json_file):
            create_embeddings_json(file)
        file = json_file

    assert get_ext(file) == '.json', 'Should now be a json file.'
    # Load the embeddings from the json file
    chapters, embeddings = process_file(file)
    if embeddings is not None:
        search(query, embeddings, file, chapters)


if __name__ == "__main__":
    book_path = \
        r'D:\Documents\Papers\EPub Books\Karl R. Popper - The Logic of Scientific Discovery-Routledge (2002).epub'
    ebook_semantic_search('Why do we need to corroborate theories at all?', book_path, do_preview=False)
