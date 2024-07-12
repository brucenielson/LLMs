from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer


class PgvectorManager:
    def __init__(self,
                 dbname='postgres',
                 user='postgres',
                 password_file=r'D:\Documents\Secrets\postgres_password.txt',
                 model_name='all-MiniLM-L6-v2'):
        self.dbname = dbname
        self.user = user
        self.password = self.get_password(password_file)
        self.model_name = model_name
        # Connect to Postgres and create the vector extension
        if self.password is None:
            raise ValueError("Failed to retrieve the database password.")
        self.conn = psycopg.connect(dbname=self.dbname, autocommit=True, user=self.user, password=self.password)
        self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)
        # Create the SentenceTransformer model
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = 512  # No idea if this works
        # see: https://huggingface.co/sentence-transformers/all-mpnet-base-v2/discussions/15
        # https://www.sbert.net/examples/applications/computing-embeddings/README.html
        # https://www.pinecone.io/learn/chunking-strategies/
        # https://medium.com/@joseph.nyirenda/mean-pooling-with-sentence-transformers-2e326f25b046
        # https://www.youtube.com/watch?v=WS1uVMGhlWQ
        # https://www.linkedin.com/pulse/prompt-engineering-chunking-strategies-ravi-naarla/
        # https://www.reddit.com/r/LocalLLaMA/comments/16cdsv6/which_sentence_transformer_is_the_best_one_for/
        # https://github.com/pgvector/pgvector-python/tree/master
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self.embeddings = None
        self.texts = None

    @staticmethod
    def get_password(password_file):
        try:
            with open(password_file, 'r') as file:
                password = file.read().strip()
        except FileNotFoundError:
            print(f"The file '{password_file}' does not exist.")
            password = None
        except Exception as e:
            print(f"An error occurred: {e}")
            password = None

        return password

    def create_embeddings(self):
        self.embeddings = self.model.encode(self.texts)
        self.vector_size = self.embeddings.shape[1]

    def create_documents_table(self):
        self.conn.execute('DROP TABLE IF EXISTS documents')
        self.conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, '
                          'embedding vector('+str(self.vector_size)+'))')

    def insert_documents(self):
        if self.embeddings is None or len(self.texts) != len(self.embeddings):
            raise ValueError("Mismatch between texts and embeddings. Ensure embeddings are created before inserting.")
        for content, embedding in zip(self.texts, self.embeddings):
            self.conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

    def find_neighbors(self, document_id=1):
        neighbors = self.conn.execute(
            'SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> '
            '(SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5',
            {'id': document_id}).fetchall()
        for neighbor in neighbors:
            print(neighbor[0])

    def store_documents(self, texts):
        self.texts = texts
        self.create_embeddings()
        self.create_documents_table()
        self.insert_documents()


def test_postgres_document_manager():
    model_name = 'sentence-transformers/all-mpnet-base-v2'  # 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    manager = PgvectorManager(model_name=model_name)
    texts = [
        'The dog is barking',
        'The cat is purring',
        'The bear is growling'
    ]
    manager.store_documents(texts)
    manager.find_neighbors(1)


if __name__ == '__main__':
    test_postgres_document_manager()
