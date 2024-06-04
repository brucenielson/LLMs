from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer


def get_password():
    secret_file = r'D:\Documents\Secrets\postgres_password.txt'
    try:
        with open(secret_file, 'r') as file:
            password = file.read()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return password


def connect_to_postgres():
    password = get_password()
    conn = psycopg.connect(dbname='postgres', autocommit=True, user='postgres', password=password)
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    return conn


def create_documents_table(conn):
    conn.execute('DROP TABLE IF EXISTS documents')
    conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')


def create_embeddings(texts, model='all-MiniLM-L6-v2'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings


def insert_documents(conn, input, embeddings):
    for content, embedding in zip(input, embeddings):
        conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))


def find_neighbors(conn, document_id=1):
    neighbors = conn.execute(
        'SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> '
        '(SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5',
        {'id': document_id}).fetchall()
    for neighbor in neighbors:
        print(neighbor[0])


def run_test():
    texts = [
        'The dog is barking',
        'The cat is purring',
        'The bear is growling'
    ]
    conn = connect_to_postgres()
    create_documents_table(conn)
    embeddings = create_embeddings(texts)
    insert_documents(conn, texts, embeddings)
    find_neighbors(conn, 1)


run_test()
