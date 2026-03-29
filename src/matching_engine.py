"""
Phase 3: Resume-Job Matching Engine
- Calculate similarity between resume and job description using TF-IDF and cosine similarity
- Return match score between 0 and 100
- Provide detailed matching analysis
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchingEngine:
    """Resume-Job Matching using TF-IDF and Cosine Similarity"""
    
    def __init__(self, nlp_processor):
        self.nlp_processor = nlp_processor
        self.tfidf_vectorizer = None
        self.resume_vectors = None
        self.job_vectors = None
    
    def calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Handle sparse matrices
        if hasattr(vec1, 'toarray'):
            vec1 = vec1.toarray()
        if hasattr(vec2, 'toarray'):
            vec2 = vec2.toarray()
        
        # Ensure proper shape
        vec1 = vec1.reshape(1, -1) if vec1.ndim == 1 else vec1
        vec2 = vec2.reshape(1, -1) if vec2.ndim == 1 else vec2
        
        similarity = cosine_similarity(vec1, vec2)
        return float(similarity[0][0])
    
    def prepare_vectors(self, resumes, jobs):
        """
        Prepare TF-IDF vectors for resumes and jobs
        
        Args:
            resumes: List of resume texts
            jobs: List of job description texts
        """
        # Combine all texts for vectorizer fitting
        all_texts = resumes + jobs
        
        self.resume_vectors, self.tfidf_vectorizer = self.nlp_processor.prepare_tfidf_features(
            all_texts,
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Split vectors back
        n_resumes = len(resumes)
        self.resume_vectors = self.resume_vectors[:n_resumes]
        self.job_vectors = self.resume_vectors[n_resumes:]
        
        return self.resume_vectors, self.job_vectors
    
    def get_text_vector(self, text):
        """Get TF-IDF vector for a given text"""
        if self.tfidf_vectorizer is None:
            # Initialize vectorizer with sample data if not already done
            logger.info("Initializing TF-IDF vectorizer with sample data...")
            sample_texts = [
                "python java javascript web development software engineer",
                "data science machine learning python r statistics",
                "project management agile scrum leadership team management",
                "devops aws docker kubernetes cloud computing",
                "ui ux design adobe photoshop figma user experience",
                "database sql mysql postgresql mongodb nosql",
                "mobile development ios android react native flutter",
                "cybersecurity network security ethical hacking penetration testing",
                "business analysis requirements gathering stakeholder management",
                "quality assurance testing automation selenium junit"
            ]
            
            self.resume_vectors, self.tfidf_vectorizer = self.nlp_processor.prepare_tfidf_features(
                sample_texts,
                max_features=5000,
                ngram_range=(1, 2)
            )
            logger.info("TF-IDF vectorizer initialized successfully")
        
        return self.tfidf_vectorizer.transform([text])
    
    def calculate_match_score(self, resume_text, job_text, normalize=True):
        """
        Calculate match score between resume and job description
        
        Args:
            resume_text: Resume text or vector
            job_text: Job description text or vector
            normalize: Whether to normalize score to 0-100
        
        Returns:
            Match score (0-100)
        """
        # Get vectors if text strings
        if isinstance(resume_text, str):
            resume_vec = self.get_text_vector(resume_text)
        else:
            resume_vec = resume_text
        
        if isinstance(job_text, str):
            job_vec = self.get_text_vector(job_text)
        else:
            job_vec = job_text
        
        # Calculate cosine similarity
        similarity = self.calculate_cosine_similarity(resume_vec, job_vec)
        
        # Normalize to 0-100 scale
        if normalize:
            score = similarity * 100
        else:
            score = similarity
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    def match_resume_to_job(self, resume_data, job_data):
        """
        Comprehensive resume-job matching with multiple scoring dimensions
        
        Args:
            resume_data: Dictionary with resume info
            job_data: Dictionary with job info
        
        Returns:
            Dictionary with detailed matching analysis
        """
        # Content similarity (TF-IDF)
        resume_text = resume_data.get('processed_text', '')
        job_text = job_data.get('job_description', '')
        
        content_score = self.calculate_match_score(resume_text, job_text)
        
        # Skill match
        resume_skills = set(resume_data.get('extracted_skills', []))
        job_skills = set(job_data.get('skills_required', []))
        
        skill_match = self._calculate_skill_match(resume_skills, job_skills)
        
        # Experience alignment
        experience_score = self._calculate_experience_alignment(
            resume_data.get('years_experience', 0),
            resume_data.get('experience_years_extracted', 0),
            job_data.get('experiencere_requirement', 0)
        )
        
        # Education alignment
        education_score = self._calculate_education_alignment(
            resume_data.get('education_level', 0),
            job_data.get('educationaL_requirements', '')
        )
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(
            content_score=content_score,
            skill_score=skill_match['match_percentage'],
            experience_score=experience_score,
            education_score=education_score,
            weights={
                'content': 0.30,
                'skill': 0.40,
                'experience': 0.20,
                'education': 0.10
            }
        )
        
        result = {
            'overall_score': overall_score,
            'content_similarity': content_score,
            'skill_match': skill_match,
            'experience_alignment': experience_score,
            'education_alignment': education_score,
            'matched_skills': skill_match['matched_skills'],
            'missing_skills': skill_match['missing_skills'],
            'recommendation': self._get_recommendation(overall_score)
        }
        
        return result
    
    @staticmethod
    def _calculate_skill_match(resume_skills, job_skills):
        """Calculate skill matching details"""
        if not job_skills:
            return {
                'matched_count': 0,
                'match_percentage': 100 if not resume_skills else 80,
                'matched_skills': list(resume_skills.intersection(job_skills)),
                'missing_skills': list(job_skills - resume_skills),
                'total_required': len(job_skills)
            }
        
        matched = resume_skills.intersection(job_skills)
        missing = job_skills - resume_skills
        
        return {
            'matched_count': len(matched),
            'match_percentage': (len(matched) / len(job_skills) * 100) if job_skills else 0,
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'total_required': len(job_skills)
        }
    
    @staticmethod
    def _calculate_experience_alignment(years_exp, extracted_exp, required_exp):
        """
        Calculate experience alignment score
        
        Returns score 0-100
        """
        try:
            # Use average of two experience estimates
            avg_experience = (years_exp + extracted_exp) / 2 if years_exp >= 0 else extracted_exp
            
            # Convert required experience (string) to number
            required = 0
            if isinstance(required_exp, str) and required_exp:
                import re
                match = re.search(r'(\d+)', str(required_exp))
                if match:
                    required = int(match.group(1))
            
            # Scoring logic
            if avg_experience >= required:
                # More experience than required is good
                excess = avg_experience - required
                score = min(100, 70 + (excess * 5))  # Bonus for extra experience
            else:
                # Less experience than required
                shortfall = required - avg_experience
                score = max(0, 70 - (shortfall * 10))  # Penalty for shortfall
            
            return score
        except:
            return 50  # Default neutral score
    
    @staticmethod
    def _calculate_education_alignment(candidate_education, job_education_req):
        """
        Calculate education alignment score
        
        Returns score 0-100
        """
        education_hierarchy = {
            'high school': 1,
            'diploma': 1,
            'bachelor': 2,
            'b.tech': 2,
            'b.sc': 2,
            'master': 3,
            'm.tech': 3,
            'm.sc': 3,
            'mba': 3,
            'phd': 4,
            'doctorate': 4
        }
        
        try:
            job_req_lower = str(job_education_req).lower()
            
            # Find minimum required education level from job description
            required_level = 1
            for edu_key, level in education_hierarchy.items():
                if edu_key in job_req_lower:
                    required_level = max(required_level, level)
            
            # Compare with candidate's education level
            candidate_level = int(candidate_education) if candidate_education else 0
            
            if candidate_level >= required_level:
                # Candidate meets or exceeds requirement
                return 90 + min(10, (candidate_level - required_level) * 5)
            else:
                # Candidate below requirement
                return max(30, 90 - (required_level - candidate_level) * 20)
        except:
            return 70  # Default score
    
    @staticmethod
    def _calculate_weighted_score(content_score, skill_score, experience_score, 
                                 education_score, weights):
        """Calculate weighted overall score"""
        weighted_score = (
            content_score * weights.get('content', 0.3) +
            skill_score * weights.get('skill', 0.4) +
            experience_score * weights.get('experience', 0.2) +
            education_score * weights.get('education', 0.1)
        )
        
        return min(100, max(0, weighted_score))  # Clamp 0-100
    
    @staticmethod
    def _get_recommendation(score):
        """Get recommendation based on score"""
        if score >= 80:
            return "STRONG_MATCH"
        elif score >= 60:
            return "GOOD_MATCH"
        elif score >= 40:
            return "MODERATE_MATCH"
        else:
            return "POOR_MATCH"
    
    def batch_match(self, resumes_df, jobs_df):
        """
        Match all resumes to all jobs
        
        Returns:
            DataFrame with match scores
        """
        results = []
        
        for idx_r, resume_row in resumes_df.iterrows():
            for idx_j, job_row in jobs_df.iterrows():
                match_result = self.match_resume_to_job(
                    resume_row.to_dict(),
                    job_row.to_dict()
                )
                
                result = {
                    'resume_index': idx_r,
                    'job_index': idx_j,
                    'job_title': job_row.get('job_position_name', 'Unknown'),
                    **match_result
                }
                results.append(result)
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed batch matching: {len(results)} comparisons")
        
        return results_df
    
    def get_top_matches(self, results_df, resume_index=None, top_n=5):
        """Get top N job matches for a resume"""
        if resume_index is not None:
            results = results_df[results_df['resume_index'] == resume_index]
        else:
            results = results_df
        
        return results.nlargest(top_n, 'overall_score')
