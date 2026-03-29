"""
Phase 6-7: Advanced Feature Engineering & Clustering
- Create advanced features: skill match score, experience weight, domain relevance
- K-Means clustering to group similar candidates
- PCA for dimensionality reduction and visualization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for improved model performance"""
    
    # Domain-specific skills mapping
    DOMAIN_SKILLS = {
        'Data Science': ['python', 'machine learning', 'data analysis', 'sql', 'tableau', 'pandas', 'scikit-learn'],
        'Web Development': ['html', 'css', 'javascript', 'react', 'angular', 'node', 'express'],
        'Cloud': ['aws', 'azure', 'gcp', 'kubernetes', 'docker', 'jenkins'],
        'DevOps': ['docker', 'kubernetes', 'jenkins', 'git', 'aws', 'terraform'],
        'Big Data': ['hadoop', 'spark', 'hive', 'mapreduce', 'scala', 'kafka']
    }
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_skill_match_score(self, resume_skills, job_skills, weight_resume=0.6):
        """
        Calculate skill match score weighted by relevance
        
        Args:
            resume_skills: List of candidate's skills
            job_skills: List of required skills
            weight_resume: Weight for matching resume skills
        
        Returns:
            Score between 0-100
        """
        if not job_skills:
            return 100 if not resume_skills else 50
        
        resume_skills_lower = {s.lower() for s in resume_skills}
        job_skills_lower = {s.lower() for s in job_skills}
        
        matched = resume_skills_lower.intersection(job_skills_lower)
        missing = job_skills_lower - resume_skills_lower
        
        # Exact matches
        match_score = (len(matched) / len(job_skills_lower)) * 100 if job_skills_lower else 0
        
        # Bonus for extra skills (shows depth)
        extra_skills = resume_skills_lower - job_skills_lower
        bonus = min(20, len(extra_skills) * 2)
        
        return min(120, match_score + bonus) / 1.2  # Normalize to 100
    
    def calculate_experience_weight(self, years_exp, required_exp, penalty_rate=10):
        """
        Calculate experience weight score
        
        Args:
            years_exp: Candidate's years of experience
            required_exp: Required years of experience
            penalty_rate: Penalty per year below requirement
        
        Returns:
            Score between 0-100
        """
        if years_exp >= required_exp:
            # Rewarding extra experience
            bonus = min(20, (years_exp - required_exp) * 5)
            return min(100, 70 + bonus)
        else:
            # Penalty for experience shortfall
            shortfall = required_exp - years_exp
            return max(10, 70 - (shortfall * penalty_rate))
    
    def calculate_education_relevance(self, candidate_education, degree_level_mapping=None):
        """
        Calculate education relevance score based on field of study
        
        Args:
            candidate_education: Education details
            degree_level_mapping: Mapping of degree types to scores
        
        Returns:
            Score between 0-100
        """
        if degree_level_mapping is None:
            degree_level_mapping = {
                'B.Tech': 60,
                'B.Sc': 60,
                'Bachelor': 60,
                'M.Tech': 80,
                'M.Sc': 80,
                'Master': 80,
                'MBA': 75,
                'PhD': 95,
                'Doctorate': 95
            }
        
        if not candidate_education:
            return 40
        
        max_score = 0
        for degree, score in degree_level_mapping.items():
            if degree.lower() in str(candidate_education).lower():
                max_score = max(max_score, score)
        
        return max_score if max_score > 0 else 40
    
    def calculate_domain_relevance(self, resume_skills, job_position):
        """
        Calculate how relevant candidate is to specific domain/role
        
        Args:
            resume_skills: Candidate's skills
            job_position: Job position title
        
        Returns:
            Score between 0-100 and matched domain
        """
        job_position_lower = str(job_position).lower()
        resume_skills_lower = {s.lower() for s in resume_skills}
        
        max_score = 0
        matched_domain = None
        
        for domain, domain_skills in self.DOMAIN_SKILLS.items():
            domain_skills_lower = {s.lower() for s in domain_skills}
            matched_domain_skills = resume_skills_lower.intersection(domain_skills_lower)
            
            domain_score = (len(matched_domain_skills) / len(domain_skills_lower)) * 100
            
            # Bonus if job title mentions domain
            if domain.lower() in job_position_lower:
                domain_score += 20
            
            if domain_score > max_score:
                max_score = domain_score
                matched_domain = domain
        
        return min(100, max_score), matched_domain
    
    def calculate_certification_score(self, certifications, job_certifications_required):
        """
        Calculate score based on relevant certifications
        
        Args:
            certifications: List of candidate's certifications
            job_certifications_required: List of required certifications
        
        Returns:
            Score between 0-100
        """
        if not job_certifications_required:
            bonus_score = min(20, len(certifications) * 5)
            return min(100, 60 + bonus_score)
        
        cert_lower = {str(c).lower() for c in certifications}
        req_lower = {str(c).lower() for c in job_certifications_required}
        
        matched = cert_lower.intersection(req_lower)
        
        return (len(matched) / len(req_lower)) * 100 if req_lower else 50
    
    def create_advanced_features(self, df, resume_job_match_df=None):
        """
        Create advanced features for improved predictions
        
        Args:
            df: DataFrame with base features
            resume_job_match_df: DataFrame with matching scores
        
        Returns:
            DataFrame with new advanced features
        """
        df = df.copy()
        
        # 1. Derived features from existing features
        df['skill_experience_ratio'] = df['num_skills'] / (df['years_experience'] + 1)
        df['role_diversity'] = df['num_positions'] / (df['num_companies'] + 1)
        df['avg_tenure'] = df['years_experience'] / (df['num_companies'] + 1)
        
        # 2. Profile completeness score
        df['profile_completeness'] = (
            (df['num_skills'] > 0).astype(int) * 25 +
            (df['education_level'] > 0).astype(int) * 25 +
            (df['years_experience'] > 0).astype(int) * 25 +
            (df['num_positions'] > 0).astype(int) * 25
        )
        
        # 3. Seniority level
        def calculate_seniority(row):
            years = row['years_experience']
            if years >= 10:
                return 4  # Senior/Lead
            elif years >= 5:
                return 3  # Mid-level
            elif years >= 2:
                return 2  # Junior
            else:
                return 1  # Entry-level
        
        df['seniority_level'] = df.apply(calculate_seniority, axis=1)
        
        # 4. Specialization score (concentration of skills in domains)
        if 'extracted_skills' in df.columns:
            df['domain_relevance_score'] = df.apply(
                lambda row: self.calculate_domain_relevance(
                    row['extracted_skills'],
                    row.get('job_position_name', '')
                )[0],
                axis=1
            )
        
        logger.info("Advanced features created successfully")
        return df
    
    def scale_features(self, df, feature_columns=None):
        """Scale numerical features"""
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_scaled = df.copy()
        df_scaled[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        return df_scaled


class CandidateClustering:
    """Cluster similar candidates using K-Means"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.pca = None
        self.cluster_labels = None
        self.silhouette_avg = None
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using Elbow method and Silhouette score
        
        Args:
            X: Feature matrix
            max_clusters: Maximum clusters to test
        
        Returns:
            Dictionary with metrics for each cluster count
        """
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        
        results = {
            'n_clusters': range(2, max_clusters + 1),
            'inertia': inertias,
            'silhouette_score': silhouette_scores,
            'davies_bouldin': davies_bouldin_scores
        }
        
        # Optimal is max silhouette score
        optimal_n = list(range(2, max_clusters + 1))[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_n}")
        
        return results, optimal_n
    
    def perform_clustering(self, X, n_clusters=None):
        """
        Perform K-Means clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (uses self.n_clusters if None)
        
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X)
        
        self.silhouette_avg = silhouette_score(X, self.cluster_labels)
        logger.info(f"Silhouette Score: {self.silhouette_avg:.4f}")
        
        return self.cluster_labels
    
    def perform_pca(self, X, n_components=2):
        """
        Perform PCA for dimensionality reduction
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep
        
        Returns:
            Reduced feature matrix
        """
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA Explained Variance: {explained_variance:.4f}")
        
        return X_reduced
    
    def visualize_clusters(self, X_pca, title="Candidate Clusters"):
        """Visualize clusters in 2D PCA space"""
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels, 
                            cmap='viridis', s=100, alpha=0.6, edgecolors='k')
        
        # Plot cluster centers
        if self.pca is not None and self.kmeans is not None:
            centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='X', s=300, edgecolors='black', linewidths=2,
                       label='Centroids')
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(title, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_cluster_stats(self, df, cluster_labels):
        """Get statistics for each cluster"""
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        stats = []
        for cluster in range(self.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            
            stat = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Avg_Skills': cluster_data['num_skills'].mean() if 'num_skills' in cluster_data else 0,
                'Avg_Experience': cluster_data['years_experience'].mean() if 'years_experience' in cluster_data else 0,
                'Avg_Seniority': cluster_data.get('seniority_level', pd.Series()).mean() if 'seniority_level' in cluster_data else 0
            }
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def plot_cluster_comparison(self, cluster_stats):
        """Plot comparison of clusters"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = ['Size', 'Avg_Skills', 'Avg_Experience']
        
        for idx, metric in enumerate(metrics):
            if metric in cluster_stats.columns:
                cluster_stats.plot(x='Cluster', y=metric, kind='bar', ax=axes[idx], legend=False)
                axes[idx].set_title(f'{metric} by Cluster', fontweight='bold')
                axes[idx].set_xlabel('Cluster')
                axes[idx].set_ylabel(metric)
                axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
