"""
FastAPI Backend for Smart Hiring Platform
Endpoints:
- /predict → candidate selection prediction
- /match-score → resume-job similarity score
- /chat → RAG chatbot response
- /skills → skill extraction
- /summary → resume summary generation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from nlp_processor import NLPProcessor, SkillMatcher
from matching_engine import MatchingEngine
from llm_integration import LLMIntegration, RAGChatbot, HiringPoliciesKnowledge
from model_persistence import ModelPersistence
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Smart Hiring Platform API",
    description="AI-powered intelligent hiring and candidate analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service components
nlp_processor = NLPProcessor()
matching_engine = MatchingEngine(nlp_processor)
llm_integration = LLMIntegration()
rag_chatbot = RAGChatbot(llm_integration)
skill_matcher = SkillMatcher(nlp_processor)
model_persistence = ModelPersistence()

# Load hiring policies into RAG
rag_chatbot.add_to_knowledge_base(
    HiringPoliciesKnowledge.get_sample_policies(),
    document_type="hiring_policy"
)

# Request/Response Models
class SkillExtractionRequest(BaseModel):
    text: str

class SkillExtractionResponse(BaseModel):
    skills: List[str]
    education: List[str]
    years_experience: int
    extracted_skills_count: int

class MatchingRequest(BaseModel):
    resume_text: str
    job_description: str

class MatchingResponse(BaseModel):
    overall_score: float
    content_similarity: float
    skill_match_percentage: float
    experience_alignment: float
    education_alignment: float
    matched_skills: List[str]
    missing_skills: List[str]
    recommendation: str

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = "Random Forest"

class PredictionResponse(BaseModel):
    prediction: int  # 0: Reject, 1: Shortlist
    confidence: float
    model_used: str

class ResumeSummaryRequest(BaseModel):
    resume_text: str

class InterviewQuestionsRequest(BaseModel):
    skills: List[str]
    job_position: str
    num_questions: Optional[int] = 5

class ChatbotRequest(BaseModel):
    message: str

class ChatbotResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None

class HiringFeedbackRequest(BaseModel):
    resume_text: str
    job_description: str
    match_score: float
    is_shortlisted: bool


# ============= ENDPOINTS =============

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "Smart Hiring Platform API",
        "version": "1.0.0"
    }


@app.post("/extract-skills", response_model=SkillExtractionResponse, tags=["NLP"])
async def extract_skills(request: SkillExtractionRequest):
    """
    Extract skills, education, and experience from text
    
    Args:
        text: Resume text or candidate information
    
    Returns:
        Extracted skills, education, and experience data
    """
    try:
        skills = nlp_processor.extract_skills_from_text(request.text)
        education = nlp_processor.extract_education(request.text)
        experience = nlp_processor.extract_years_of_experience(request.text)
        
        return SkillExtractionResponse(
            skills=skills,
            education=education,
            years_experience=experience,
            extracted_skills_count=len(skills)
        )
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-score", response_model=MatchingResponse, tags=["Matching"])
async def calculate_match_score(request: MatchingRequest):
    """
    Calculate resume-job match score using TF-IDF and skill matching
    
    Args:
        resume_text: Candidate's resume
        job_description: Job description
    
    Returns:
        Detailed matching analysis with scores
    """
    try:
        # Preprocess texts
        processed_resume = nlp_processor.preprocess_text(request.resume_text)
        processed_job = nlp_processor.preprocess_text(request.job_description)
        
        # Extract skills and features
        resume_data = {
            'processed_text': processed_resume,
            'extracted_skills': nlp_processor.extract_skills_from_text(request.resume_text),
            'years_experience': nlp_processor.extract_years_of_experience(request.resume_text),
            'education_level': 2
        }
        
        job_data = {
            'job_description': processed_job,
            'skills_required': nlp_processor.extract_skills_from_text(request.job_description),
            'experiencere_requirement': 0,
            'educationaL_requirements': ''
        }
        
        # Calculate match
        result = matching_engine.match_resume_to_job(resume_data, job_data)
        
        return MatchingResponse(
            overall_score=result['overall_score'],
            content_similarity=result['content_similarity'],
            skill_match_percentage=result['skill_match']['match_percentage'],
            experience_alignment=result['experience_alignment'],
            education_alignment=result['education_alignment'],
            matched_skills=result['matched_skills'],
            missing_skills=result['missing_skills'],
            recommendation=result['recommendation']
        )
    except Exception as e:
        logger.error(f"Error calculating match score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["ML Models"])
async def predict_candidate_selection(request: PredictionRequest):
    """
    Predict whether candidate should be shortlisted
    
    Args:
        features: List of numerical features
        model_name: ML model to use
    
    Returns:
        Prediction (0: Reject, 1: Shortlist) and confidence
    """
    try:
        # Load model
        try:
            model = model_persistence.load_model(request.model_name)
        except:
            logger.warning(f"Model {request.model_name} not found. Using default.")
            return PredictionResponse(
                prediction=0,
                confidence=0.0,
                model_used="model_not_found"
            )
        
        # Make prediction
        import numpy as np
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.5
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
            model_used=request.model_name
        )
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume-summary", tags=["LLM"])
async def generate_resume_summary(request: ResumeSummaryRequest):
    """
    Generate professional summary from resume
    
    Args:
        resume_text: Resume content
    
    Returns:
        Generated summary
    """
    try:
        summary = llm_integration.generate_resume_summary(request.resume_text)
        return {
            "summary": summary,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview-questions", tags=["LLM"])
async def generate_interview_questions(request: InterviewQuestionsRequest):
    """
    Generate interview questions based on skills and position
    
    Args:
        skills: List of candidate skills
        job_position: Target job position
        num_questions: Number of questions to generate
    
    Returns:
        List of interview questions
    """
    try:
        questions = llm_integration.generate_interview_questions(
            request.skills,
            request.job_position,
            request.num_questions
        )
        return {
            "questions": questions,
            "count": len(questions),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hiring-feedback", tags=["LLM"])
async def generate_hiring_feedback(request: HiringFeedbackRequest):
    """
    Generate hiring decision feedback
    
    Args:
        resume_text: Candidate resume
        job_description: Job description
        match_score: Resume-job match score
        is_shortlisted: Whether candidate is shortlisted
    
    Returns:
        Hiring feedback
    """
    try:
        feedback = llm_integration.generate_hiring_feedback(
            request.resume_text,
            request.job_description,
            request.match_score,
            request.is_shortlisted
        )
        return {
            "feedback": feedback,
            "decision": "SHORTLISTED" if request.is_shortlisted else "REJECTED",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatbotResponse, tags=["RAG Chatbot"])
async def chatbot(request: ChatbotRequest):
    """
    RAG-based chatbot for recruiter queries
    
    Args:
        message: User's question
    
    Returns:
        Chatbot response
    """
    try:
        response = rag_chatbot.chat(request.message)
        return ChatbotResponse(
            response=response,
            conversation_id="default"
        )
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Models"])
async def list_available_models():
    """List all available trained models"""
    try:
        models = model_persistence.list_models()
        return {
            "models": models,
            "total": sum(len(v) for v in models.values()),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "nlp_ready": True,
        "llm_configured": llm_integration.client is not None,
        "rag_chatbot_ready": True,
        "models_directory": "models"
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": str(exc),
        "status": "error"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
