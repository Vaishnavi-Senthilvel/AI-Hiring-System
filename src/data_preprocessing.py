"""
Phase 1: Data Preprocessing & Feature Extraction
- Load resume and job description datasets
- Clean missing values and remove duplicates
- Extract features like years of experience and number of skills
- Encode categorical variables and scale numerical features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import ast
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, cleaning, and feature extraction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_data(self, filepath):
        """Load resume and job data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_missing_values(self, df, threshold=0.5):
        """
        Remove rows with excessive missing values
        
        Args:
            df: DataFrame to clean
            threshold: Proportion of missing values to drop a row
        
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        df = df.dropna(thresh=len(df.columns) * (1 - threshold))
        logger.info(f"Removed {initial_rows - len(df)} rows with excessive missing values")
        
        # Fill remaining missing values
        df = df.fillna({
            'skills': '[]',
            'career_objective': 'Not specified',
            'degrees': '',
            'related_skils_in_job': '[]',
            'skills_required': '[]'
        })
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate records"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        return df
    
    def parse_list_columns(self, df):
        """Convert string representations of lists to actual lists"""
        list_columns = ['skills', 'related_skils_in_job', 'skills_required']
        
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._safe_eval_list(x))
        
        return df
    
    @staticmethod
    def _safe_eval_list(x):
        """Safely evaluate string representation of list"""
        if isinstance(x, list):
            return x
        if pd.isna(x) or x == '[]' or x == '' or x is None:
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []
    
    def extract_features(self, df):
        """Extract meaningful features from resume data"""
        
        # Number of skills
        df['num_skills'] = df['skills'].apply(lambda x: len(x) if x else 0)
        
        # Years of experience (from start_dates and end_dates)
        df['years_experience'] = self._calculate_years_experience(df)
        
        # Number of positions held
        df['num_positions'] = df['positions'].apply(
            lambda x: len(self._safe_eval_list(x)) if pd.notna(x) else 0
        )
        
        # Number of companies worked
        df['num_companies'] = df['professional_company_names'].apply(
            lambda x: len(self._safe_eval_list(x)) if pd.notna(x) else 0
        )
        
        # Education level encoding
        df['education_level'] = df['degree_names'].apply(self._encode_education_level)
        
        # Skills match (for job matching)
        if 'skills_required' in df.columns:
            df['skill_match_count'] = df.apply(
                lambda row: self._count_matched_skills(row['skills'], row['skills_required']),
                axis=1
            )
        
        logger.info("Feature extraction completed")
        return df
    
    @staticmethod
    def _calculate_years_experience(df):
        """Calculate years of experience from start and end dates"""
        try:
            if 'start_dates' not in df.columns or 'end_dates' not in df.columns:
                return pd.Series([0] * len(df))
            
            # Parse dates and calculate duration
            years = []
            for _, row in df.iterrows():
                start_dates = row.get('start_dates', [])
                end_dates = row.get('end_dates', [])
                
                start_list = DataPreprocessor._safe_eval_list(start_dates)
                end_list = DataPreprocessor._safe_eval_list(end_dates)
                
                total_years = 0
                for start, end in zip(start_list, end_list):
                    # Simple year extraction
                    try:
                        start_year = int(str(start)[-4:])
                        end_year = int(str(end)[-4:]) if 'Till' not in str(end) else 2024
                        total_years += max(0, end_year - start_year)
                    except:
                        continue
                
                years.append(total_years)
            
            return pd.Series(years)
        except Exception as e:
            logger.warning(f"Error calculating experience: {e}")
            return pd.Series([0] * len(df))
    
    @staticmethod
    def _encode_education_level(degrees):
        """Encode education level numerically"""
        degree_hierarchy = {
            'B.Tech': 2,
            'B.Sc': 2,
            'Bachelor': 2,
            'M.Tech': 3,
            'M.Sc': 3,
            'Master': 3,
            'MBA': 3,
            'PhD': 4,
            'Doctorate': 4
        }
        
        if pd.isna(degrees):
            return 0
        
        degree_list = DataPreprocessor._safe_eval_list(degrees)
        if not degree_list:
            return 0
        
        max_level = 0
        for degree in degree_list:
            for key, value in degree_hierarchy.items():
                if key.lower() in str(degree).lower():
                    max_level = max(max_level, value)
        
        return max_level
    
    @staticmethod
    def _count_matched_skills(resume_skills, job_skills):
        """Count how many resume skills match job requirements"""
        if not resume_skills or not job_skills:
            return 0
        
        resume_skills_lower = {str(s).lower() for s in resume_skills}
        job_skills_lower = {str(s).lower() for s in job_skills}
        
        return len(resume_skills_lower.intersection(job_skills_lower))
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """Encode categorical variables"""
        if categorical_cols is None:
            categorical_cols = []
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_numerical_features(self, df, numerical_cols=None):
        """Scale numerical features using StandardScaler"""
        if numerical_cols is None:
            numerical_cols = ['num_skills', 'years_experience', 'num_positions', 
                            'num_companies', 'education_level', 'skill_match_count']
        
        available_cols = [col for col in numerical_cols if col in df.columns]
        self.feature_columns = available_cols
        
        df[available_cols] = self.scaler.fit_transform(df[available_cols])
        logger.info(f"Scaled {len(available_cols)} numerical features")
        
        return df
    
    def preprocess_pipeline(self, filepath):
        """
        Complete preprocessing pipeline
        
        Returns:
            Preprocessed DataFrame with new features
        """
        # Load
        df = self.load_data(filepath)
        
        # Clean
        df = self.clean_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.parse_list_columns(df)
        
        # Extract features
        df = self.extract_features(df)
        
        # Encode and scale
        df = self.encode_categorical_features(df, categorical_cols=['major_field_of_studies', 'job_position_name'])
        df = self.scale_numerical_features(df)
        
        logger.info("Preprocessing pipeline completed")
        return df


def split_candidate_job_data(df):
    """
    Split candidate resume data from job requirements
    Returns: resume_df, job_df
    """
    # Assuming rows with career_objective are candidates
    # and rows with job_position_name are job postings
    
    candidate_cols = ['career_objective', 'professional_company_names', 'positions']
    job_cols = ['job_position_name', 'skills_required', 'experiencere_requirement']
    
    has_career_obj = df['career_objective'].notna() & (df['career_objective'] != '')
    
    candidate_df = df[has_career_obj].copy()
    job_df = df[~has_career_obj].copy()
    
    logger.info(f"Split data: {len(candidate_df)} candidates, {len(job_df)} jobs")
    
    return candidate_df, job_df
