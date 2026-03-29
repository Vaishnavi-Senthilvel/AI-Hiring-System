"""
Phase 10-12: LLM Integration, RAG Chatbot & FastAPI Endpoints
- Resume summary generation using LLM
- Interview questions generation
- Hiring feedback
- RAG-based chatbot for recruiter queries
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMIntegration:
    """LLM-based feature generation"""
    
    def __init__(self, api_key=None):
        """
        Initialize LLM integration
        
        Args:
            api_key: OpenAI API key (uses env var if None)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. LLM features will be limited.")
            self.client = None
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI package not installed. Install with: pip install openai")
                self.client = None
    
    def generate_resume_summary(self, resume_text: str, max_length: int = 200) -> str:
        """
        Generate professional summary from resume
        
        Args:
            resume_text: Resume content
            max_length: Maximum summary length
        
        Returns:
            Professional summary
        """
        if not self.client:
            return "LLM service not available. Please configure OpenAI API key."
        
        try:
            prompt = f"""
            Based on the following resume, generate a professional 1-2 sentence summary highlighting 
            the candidate's key strengths and expertise. Keep it concise and impactful.
            
            Resume:
            {resume_text}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR professional."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary at this time."
    
    def generate_interview_questions(self, candidate_skills: List[str], 
                                    job_position: str, num_questions: int = 5) -> List[str]:
        """
        Generate interview questions based on candidate skills and job position
        
        Args:
            candidate_skills: List of candidate's skills
            job_position: Job position/title
            num_questions: Number of questions to generate
        
        Returns:
            List of interview questions
        """
        if not self.client:
            return ["LLM service not available. Please configure OpenAI API key."]
        
        try:
            skills_str = ', '.join(candidate_skills[:10])  # Use top 10 skills
            
            prompt = f"""
            Generate {num_questions} technical interview questions for a {job_position} position.
            The candidate has the following skills: {skills_str}
            
            Format each question on a new line, numbered.
            Focus on practical, scenario-based questions that assess competency.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8
            )
            
            # Parse questions
            text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in text.split('\n') if q.strip()]
            
            return questions[:num_questions]
        
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return [f"Error generating questions: {str(e)}"]
    
    def generate_hiring_feedback(self, resume_text: str, job_description: str,
                                match_score: float, is_shortlisted: bool) -> str:
        """
        Generate hiring feedback explaining decision
        
        Args:
            resume_text: Resume content
            job_description: Job description
            match_score: Resume-job match score (0-100)
            is_shortlisted: Whether candidate is shortlisted
        
        Returns:
            Hiring feedback
        """
        if not self.client:
            return "LLM service not available. Please configure OpenAI API key."
        
        try:
            decision = "SHORTLISTED" if is_shortlisted else "REJECTED"
            
            prompt = f"""
            As an HR professional, provide constructive hiring feedback for a candidate.
            
            Decision: {decision}
            Match Score: {match_score:.1f}/100
            
            Resume Highlights:
            {resume_text[:500]}
            
            Job Requirements:
            {job_description[:500]}
            
            Generate 2-3 sentences explaining the hiring decision, highlighting:
            - Key strengths or gaps
            - Recommendations for the candidate
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR consultant providing constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return "Unable to generate feedback at this time."


class RAGChatbot:
    """RAG-based chatbot for recruiter queries"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """
        Initialize RAG chatbot
        
        Args:
            llm_integration: LLMIntegration instance
        """
        self.llm = llm_integration
        self.knowledge_base = []
        self.conversation_history = []
        
        # Initialize embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.using_embeddings = True
        except ImportError:
            logger.warning("sentence-transformers not installed. Using keyword matching.")
            self.embedder = None
            self.using_embeddings = False
    
    def add_to_knowledge_base(self, documents: List[str], document_type: str = "general"):
        """
        Add documents to knowledge base
        
        Args:
            documents: List of text documents
            document_type: Type of documents (policies, guidelines, etc.)
        """
        for doc in documents:
            self.knowledge_base.append({
                'content': doc,
                'type': document_type,
                'embedding': self.embedder.encode(doc) if self.using_embeddings else None
            })
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents based on query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            List of relevant document snippets
        """
        if not self.knowledge_base:
            return []
        
        if self.using_embeddings:
            # Semantic search using embeddings
            query_embedding = self.embedder.encode(query)
            
            # Calculate similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = []
            for doc in self.knowledge_base:
                if doc['embedding'] is not None:
                    sim = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
                    similarities.append(sim)
                else:
                    similarities.append(0)
            
            # Get top-k
            import numpy as np
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [self.knowledge_base[i]['content'] for i in top_indices if similarities[i] > 0.3]
        else:
            # Keyword-based retrieval
            query_lower = query.lower()
            relevant = []
            
            for doc in self.knowledge_base:
                doc_lower = doc['content'].lower()
                if any(word in doc_lower for word in query_lower.split()):
                    relevant.append(doc['content'])
            
            return relevant[:top_k]
    
    def chat(self, user_query: str) -> str:
        """
        Get chatbot response to user query
        
        Args:
            user_query: User's question
        
        Returns:
            Chatbot response
        """
        # Retrieve relevant context
        context = self.retrieve_relevant_context(user_query)
        
        # Build context string
        context_str = "\n".join(context) if context else "No relevant information found in knowledge base."
        
        # Prepare prompt
        prompt = f"""
        You are an expert HR assistant helping recruiters with hiring process questions.
        
        Relevant company information:
        {context_str}
        
        User Question: {user_query}
        
        Provide a helpful, accurate answer based on the information provided.
        """
        
        try:
            response = self.llm.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            ) if self.llm.client else None
            
            if response:
                answer = response.choices[0].message.content.strip()
            else:
                answer = "Assistant is not available. Please configure LLM settings."
        
        except Exception as e:
            logger.error(f"Error in chatbot: {e}")
            answer = f"Error processing query: {str(e)}"
        
        # Store in conversation history
        self.conversation_history.append({
            'user': user_query,
            'assistant': answer
        })
        
        return answer
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class HiringPoliciesKnowledge:
    """Sample hiring policies and guidelines for RAG"""
    
    @staticmethod
    def get_sample_policies() -> List[str]:
        """Get sample hiring policies"""
        return [
            """
            Equal Opportunity Policy: We are committed to providing equal employment opportunities to all individuals 
            regardless of race, color, religion, gender, age, national origin, disability, or sexual orientation.
            """,
            """
            Diversity & Inclusion: We actively recruit from diverse backgrounds and maintain an inclusive hiring process. 
            We encourage applications from underrepresented groups in technology.
            """,
            """
            Background Check Policy: All candidates must pass background verification before finalization. 
            We conduct verification of educational credentials and previous employment.
            """,
            """
            Salary Bands: Entry-level (0-2 years): $60k-$80k, Junior (2-5 years): $80k-$120k, 
            Mid-level (5-10 years): $120k-$180k, Senior (10+ years): $180k-$250k+
            """,
            """
            Benefits Package: All full-time employees receive health insurance, 401(k) matching, 
            unlimited PTO, professional development budget, and flexible working arrangements.
            """,
            """
            Interview Process: Standard process includes: Initial screening (30 min), Technical assessment (60 min), 
            System design round (60 min), Behavioral interview (45 min), Manager round (45 min).
            """,
            """
            Skills Requirements: For Data Science roles: Python/R, Machine Learning, SQL, Statistics, 
            Data Visualization. For Full Stack: JavaScript, React, Node.js, Databases, REST APIs.
            """
        ]
