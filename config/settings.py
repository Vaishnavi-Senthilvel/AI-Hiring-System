import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
PINECONE_ENV = os.getenv('PINECONE_ENV', 'us-west2-e1')

# Model Configuration
MODEL_NAME = 'gpt-3.5-turbo'
EMBEDDING_MODEL = 'text-embedding-ada-002'
CANDIDATE_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# NLP Configuration
STOPWORDS_LANGUAGE = 'english'
VECTORIZER_MAX_FEATURES = 5000
VECTORIZER_NGRAM_RANGE = (1, 2)

# ML Model Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Matching Engine
MIN_MATCH_SCORE = 40.0
MAX_MATCH_SCORE = 100.0

# Clustering
N_CLUSTERS = 5
PCA_COMPONENTS = 2
