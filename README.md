# 🚀 AI-Powered Smart Hiring & Candidate Intelligence Platform

A comprehensive end-to-end machine learning and AI solution for intelligent candidate selection, resume analysis, and hiring optimization.

## 🎯 Project Overview

This platform leverages cutting-edge AI technologies to automate and enhance the hiring process:

- **Resume Parsing & Analysis** - Intelligent extraction of skills, experience, and education
- **NLP Processing** - Advanced text preprocessing with tokenization, lemmatization, and skill extraction
- **Intelligent Matching** - TF-IDF + cosine similarity for resume-job alignment (0-100 scoring)
- **Predictive Analytics** - ML models to predict candidate selection probability
- **Clustering & Segmentation** - Group similar candidates using K-Means and PCA visualization
- **LLM Integration** - Generate resume summaries, interview questions, and hiring feedback
- **RAG Chatbot** - Retrieval-Augmented Generation for recruiter Q&A
- **FastAPI Backend** - Production-ready REST API endpoints
- **Streamlit UI** - Interactive web interface for recruiters

---

## 📁 Project Structure

```
├── resume_data.csv                 # Sample dataset
├── requirements.txt                # Python dependencies
├── config/
│   └── settings.py                # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Phase 1: Data cleaning & features
│   ├── nlp_processor.py            # Phase 2: NLP & skill extraction
│   ├── matching_engine.py          # Phase 3: Resume-job matching
│   ├── ml_models.py                # Phase 4-5: ML model training & evaluation
│   ├── feature_engineering.py      # Phase 6-7: Advanced features & clustering
│   ├── model_persistence.py        # Phase 8: Model saving/loading
│   └── llm_integration.py          # Phase 10-12: LLM & RAG chatbot
├── notebooks/
│   └── master_pipeline.ipynb       # Complete end-to-end pipeline
├── models/                         # Trained models (auto-created)
├── data/                           # Data directory
├── app.py                          # Phase 9: Streamlit web app
└── api.py                          # FastAPI backend
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM (Optional)

```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Run Master Pipeline Notebook

```bash
jupyter notebook notebooks/master_pipeline.ipynb
```

### 4. Launch Streamlit Web App

```bash
streamlit run app.py
```

### 5. Start FastAPI Server

```bash
python api.py
```

FastAPI will be available at: **http://localhost:8000**  
API Documentation: **http://localhost:8000/docs**

---

## 📊 Features & Phases

### Phase 1: Data Preprocessing ✅
- Load CSV data with pandas
- Clean missing values and remove duplicates
- Extract features: years of experience, skills count, education level
- Encode categorical variables
- Scale numerical features using StandardScaler

### Phase 2: NLP & Skill Extraction ✅
- Tokenization and lemmatization with NLTK
- Stopword removal
- Skill extraction from resume text
- Education and experience extraction
- TF-IDF vectorization (1-2 grams)

### Phase 3: Matching Engine ✅
- Calculate TF-IDF + cosine similarity
- Skill matching algorithm
- Experience alignment scoring
- Education relevance scoring
- Weighted overall score (0-100)

### Phase 4-5: ML Model Training & Evaluation ✅
**Models Trained:**
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

**Features:**
- GridSearchCV for hyperparameter tuning
- 5-fold cross-validation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Model comparison and best model selection

### Phase 6-7: Feature Engineering & Clustering ✅
**Advanced Features:**
- Skill-experience ratio
- Role diversity score
- Average tenure
- Profile completeness
- Seniority level
- Domain relevance

**Clustering:**
- K-Means with optimal cluster detection
- PCA for 2D visualization
- Silhouette score analysis
- Cluster statistics

### Phase 8: Model Persistence ✅
- Save/load models with joblib
- Version management
- Scaler and vectorizer persistence
- Pipeline state tracking

### Phase 9: Streamlit Web App ✅
**Features:**
- Dashboard with metrics
- Resume upload and analysis
- Real-time resume-job matching
- Skill extraction display
- Match score visualization
- Settings configuration

### Phase 10-12: LLM Integration ✅
**LLM Features:**
- Resume summary generation
- Interview questions (5 questions)
- Hiring feedback and recommendations
- RAG-based recruiter chatbot

**FastAPI Endpoints:**
- `POST /extract-skills` - Extract skills from text
- `POST /match-score` - Calculate match score
- `POST /predict` - ML prediction
- `POST /resume-summary` - Generate summary
- `POST /interview-questions` - Generate questions
- `POST /hiring-feedback` - Generate feedback
- `POST /chat` - Chatbot response
- `GET /models` - List available models
- `GET /health` - Health check

---

## 🔑 Key Technologies

| Layer | Technology |
|-------|-----------|
| **Data Processing** | pandas, numpy |
| **NLP** | NLTK, scikit-learn TfidfVectorizer |
| **Machine Learning** | scikit-learn (5 models), GridSearchCV |
| **LLM** | OpenAI GPT-3.5-turbo, sentence-transformers |
| **Web Framework** | Streamlit (UI), FastAPI (API) |
| **Visualization** | matplotlib, seaborn |
| **Vector DB** | FAISS/Pinecone-ready |
| **Model Persistence** | joblib, pickle |

---

## 📈 Expected Performance

### ML Models (typical):
- **Accuracy**: 75-90%
- **F1-Score**: 0.72-0.88
- **ROC-AUC**: 0.80-0.95

### Matching Engine:
- **Match Score Range**: 0-100
- **Classification**: Strong/Good/Moderate/Poor Match

### Clustering:
- **Silhouette Score**: 0.40-0.70
- **Optimal Clusters**: 3-7 (data-dependent)

---

## 🎯 Usage Examples

### Example 1: Calculate Match Score

```python
from src.matching_engine import MatchingEngine
from src.nlp_processor import NLPProcessor

nlp = NLPProcessor()
matcher = MatchingEngine(nlp)

result = matcher.match_resume_to_job(
    resume_data={
        'processed_text': 'Python developer with 5 years...',
        'extracted_skills': ['Python', 'Django', 'AWS'],
        'years_experience': 5
    },
    job_data={
        'job_description': 'Looking for Python engineer...',
        'skills_required': ['Python', 'Django', 'PostgreSQL']
    }
)

print(f"Match Score: {result['overall_score']:.1f}/100")
```

### Example 2: Train ML Models

```python
from src.ml_models import MLModelTrainer

trainer = MLModelTrainer()
trainer.prepare_data(X, y)
trainer.train_all_models()
results = trainer.evaluate_all_models()
print(results)
```

### Example 3: Extract Skills

```python
from src.nlp_processor import NLPProcessor

nlp = NLPProcessor()
skills = nlp.extract_skills_from_text(resume_text)
education = nlp.extract_education(resume_text)
experience = nlp.extract_years_of_experience(resume_text)
```

---

## 📝 API Examples

### Extract Skills
```bash
curl -X POST http://localhost:8000/extract-skills \
  -H "Content-Type: application/json" \
  -d '{"text": "Python developer with AWS and Django experience"}'
```

### Calculate Match Score
```bash
curl -X POST http://localhost:8000/match-score \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "5 years Python developer...",
    "job_description": "Senior Python engineer needed..."
  }'
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [8.5, 5, 3, 2, 2],
    "model_name": "Random Forest"
  }'
```

### Chatbot Query
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are our salary bands for senior roles?"}'
```

---

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# API Keys
OPENAI_API_KEY = 'your-key-here'

# Model Settings
TEST_SIZE = 0.2
CV_FOLDS = 5
N_CLUSTERS = 5

# Matching Engine
MIN_MATCH_SCORE = 40.0
MAX_MATCH_SCORE = 100.0

# NLP
VECTORIZER_MAX_FEATURES = 5000
VECTORIZER_NGRAM_RANGE = (1, 2)
```

---

## 📊 Sample Dashboard Views

### Resume Analysis
- Extracted skills
- Years of experience
- Education details
- Skill count summary

### Matching Results
- Overall match score (0-100)
- Content similarity
- Skill match percentage  
- Experience alignment
- Matched vs missing skills

### Analytics
- Skill demand analytics
- Top 20 in-demand skills
- Candidate distribution by experience
- Model performance comparison

---

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Heroku/Cloud Deployment
1. Set environment variables (OPENAI_API_KEY, etc.)
2. Configure Procfile for FastAPI or Streamlit
3. Deploy using git push

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| NLTK data missing | Run: `python -m nltk.downloader punkt stopwords wordnet` |
| OpenAI API errors | Check API key in .env and account credits |
| Memory issues | Reduce MAX_FEATURES or use subset of data |
| Port already in use | Change port: `streamlit run app.py --server.port 8501` |

---

## 📚 References & Documentation

- **scikit-learn**: https://scikit-learn.org/
- **NLTK**: https://www.nltk.org/
- **Streamlit**: https://docs.streamlit.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenAI**: https://platform.openai.com/docs/

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Advanced resume parsing (PDF/DOCX support)
- Multi-language NLP support
- Real-time model retraining pipeline
- Advanced visualization dashboard
- Integration with ATS systems

---

## 📄 License

MIT License - Free for academic and commercial use

---

## 👤 Author

Vaishnavi S  
Date: 2024  
Email: contact@example.com

---

## 🙏 Acknowledgments

- NLTK and scikit-learn communities
- OpenAI for LLM capabilities
- Streamlit and FastAPI teams
- All contributors and testers

---

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review notebook examples
3. Check API documentation (http://localhost:8000/docs)
4. Create an issue in the repository

---

**🎉 Happy Hiring! Make better hiring decisions with AI.** 🚀
