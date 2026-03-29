"""
Phase 2: NLP & Skill Extraction
- Tokenization and text preprocessing
- Stopword removal and lemmatization
- Extract skills, education, and experience from text
- Convert text into TF-IDF vectors with n-grams
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPProcessor:
    """Handles NLP preprocessing and skill extraction"""
    
    # Common technical skills database
    COMMON_SKILLS = {
        'programming': ['python', 'java', 'c++', 'javascript', 'c#', 'php', 'swift', 'kotlin', 'go', 'rust', 'ruby', 'perl', 'scala', 'r', 'matlab'],
        'web': ['html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'fastapi', 'spring', 'laravel', 'asp.net', 'jquery', 'bootstrap'],
        'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'elasticsearch', 'oracle', 'sqlite', 'firebase', 'dynamodb'],
        'cloud': ['aws', 'azure', 'gcp', 'heroku', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'terraform', 'ansible', 'puppet', 'chef'],
        'data': ['apache', 'spark', 'hadoop', 'hive', 'pig', 'mapreduce', 'hbase', 'flume', 'sqoop', 'kafka', 'airflow', 'tableau', 'power bi'],
        'ml': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy', 'matplotlib', 'seaborn'],
        'mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'cordova', 'swift', 'kotlin', 'objective-c'],
        'tools': ['git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'trello', 'asana', 'postman', 'swagger', 'vscode', 'intellij', 'eclipse'],
        'soft_skills': ['communication', 'leadership', 'teamwork', 'problem-solving', 'project management', 'agile', 'scrum', 'kanban']
    }
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.skill_keywords = self._flatten_skills_db()
    
    @staticmethod
    def _flatten_skills_db():
        """Flatten skills database into single set"""
        all_skills = set()
        for category in NLPProcessor.COMMON_SKILLS.values():
            all_skills.update(category)
        return all_skills
    
    def preprocess_text(self, text):
        """
        Complete text preprocessing pipeline
        - Lowercase
        - Remove special characters
        - Tokenization
        - Stopword removal
        - Lemmatization
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep some important ones)
        text = re.sub(r'[^a-zA-Z\s\+\#\-\.]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    
    def extract_skills_from_text(self, text):
        """Extract skills from resume text"""
        if not isinstance(text, str) or pd.isna(text):
            return []
        
        text_lower = text.lower()
        extracted_skills = []
        
        # Check each known skill
        for skill in self.skill_keywords:
            if skill in text_lower:
                # Avoid false matches (e.g., 'java' in 'javascript')
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_skills.append(skill)
        
        return list(set(extracted_skills))  # Remove duplicates
    
    def extract_education(self, text):
        """Extract education information from text"""
        if not isinstance(text, str) or pd.isna(text):
            return []
        
        text_lower = text.lower()
        education_keywords = [
            'b.tech', 'b.sc', 'bachelor', 'b.e', 'b.a', 'b.com', 'b.ba', 'b.c.a', 'b.b.a',
            'm.tech', 'm.sc', 'master', 'm.e', 'm.a', 'mba', 'm.com', 'm.c.a', 'm.s', 'm.phil',
            'phd', 'doctorate', 'ph.d', 'dphil', 'postgraduate', 'graduate', 'undergraduate',
            'university', 'college', 'institute', 'school', 'academy', 'engineering', 'technology',
            'computer science', 'information technology', 'software engineering', 'electrical engineering',
            'mechanical engineering', 'civil engineering', 'business administration', 'commerce'
        ]
        
        found_education = []
        for keyword in education_keywords:
            if keyword in text_lower:
                found_education.append(keyword.title())  # Capitalize for display
        
        return list(set(found_education))
    
    def extract_years_of_experience(self, text):
        """Extract years of experience from text"""
        if not isinstance(text, str) or pd.isna(text):
            return 0
        
        # Look for various patterns
        patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\s*-\s*(\d+)\s+years',
            r'experience\s*(?:of\s*)?(\d+)\s*\+?\s*years',
            r'(\d+)\s*years?\s*(?:of\s*)?experience',
            r'worked\s*for\s*(\d+)\s*years',
            r'(\d+)\+?\s*years?\s*in\s*',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    # Handle range patterns like "3-5 years"
                    years.extend([int(x) for x in match if x.isdigit()])
                else:
                    if match.isdigit():
                        years.append(int(match))
        
        return max(years) if years else 0
    
    def prepare_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Create TF-IDF vectorizer and transform texts
        
        Args:
            texts: List of text documents
            max_features: Maximum number of features
            ngram_range: N-gram range (1, 2) for unigrams and bigrams
        
        Returns:
            TF-IDF matrix and vectorizer
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
        
        return tfidf_matrix, self.tfidf_vectorizer
    
    def get_feature_names(self):
        """Get TF-IDF feature names"""
        if self.tfidf_vectorizer is None:
            return []
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def process_resume_data(self, df, text_column='career_objective'):
        """
        Process resume data with complete NLP pipeline
        
        Args:
            df: DataFrame with resume data
            text_column: Column name containing resume text
        
        Returns:
            DataFrame with extracted features
        """
        logger.info("Processing resume data through NLP pipeline...")
        
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Extract features
        df['extracted_skills'] = df[text_column].apply(self.extract_skills_from_text)
        df['education'] = df[text_column].apply(self.extract_education)
        df['experience_years_extracted'] = df[text_column].apply(self.extract_years_of_experience)
        
        # Add skill count feature
        df['extracted_skills_count'] = df['extracted_skills'].apply(len)
        
        logger.info("NLP processing completed")
        return df
    
    def get_top_skills(self, df, top_n=20):
        """Get most common skills across all resumes"""
        all_skills = []
        for skills_list in df['extracted_skills']:
            all_skills.extend(skills_list)
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        
        return skill_counts.most_common(top_n)
    
    def create_skill_vocabulary(self, df, min_frequency=2):
        """Create vocabulary of skills from resume data"""
        from collections import Counter
        
        all_skills = []
        for skills_list in df['extracted_skills']:
            all_skills.extend(skills_list)
        
        skill_counts = Counter(all_skills)
        
        # Filter by minimum frequency
        vocabulary = {
            skill: idx
            for idx, (skill, count) in enumerate(
                sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
            )
            if count >= min_frequency
        }
        
        logger.info(f"Created skill vocabulary with {len(vocabulary)} unique skills")
        return vocabulary


class SkillMatcher:
    """Match resume skills with job requirements"""
    
    def __init__(self, nlp_processor):
        self.nlp_processor = nlp_processor
    
    def calculate_skill_match(self, resume_skills, job_skills):
        """
        Calculate skill match score between resume and job
        
        Returns:
            (matched_count, match_percentage, matched_skills, missing_skills)
        """
        resume_skills_lower = {s.lower() for s in resume_skills}
        job_skills_lower = {s.lower() for s in job_skills}
        
        matched = resume_skills_lower.intersection(job_skills_lower)
        missing = job_skills_lower - resume_skills_lower
        
        match_percentage = (len(matched) / len(job_skills_lower) * 100) if job_skills_lower else 0
        
        return {
            'matched_count': len(matched),
            'match_percentage': match_percentage,
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'total_required': len(job_skills_lower)
        }
    
    def find_similar_skills(self, skill, resume_skills, similarity_threshold=0.8):
        """Find similar skills using string similarity"""
        from difflib import SequenceMatcher
        
        similar = []
        for resume_skill in resume_skills:
            similarity = SequenceMatcher(None, skill.lower(), resume_skill.lower()).ratio()
            if similarity >= similarity_threshold:
                similar.append((resume_skill, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
