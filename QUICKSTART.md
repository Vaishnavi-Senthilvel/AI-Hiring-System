# 🚀 QUICK START GUIDE

## Smart Hiring & Candidate Intelligence Platform

**Complete setup guide - 5 minutes to production-ready system**

---

## ⚡ Installation Steps

### Step 1: Install Python Dependencies
```bash
# From project root directory
pip install -r requirements.txt
```

**Expected Time:** 2-3 minutes

---

### Step 2: Configure Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env and add your keys (optional for basic features)
# Required only for LLM features:
# - OPENAI_API_KEY=your-key
# - PINECONE_API_KEY=your-key
```

**Note:** All features except LLM summary/questions/feedback will work without API keys

---

### Step 3: Download NLTK Data

```python
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print('✅ NLTK data downloaded successfully!')
"
```

---

## 🎯 Running the Platform

### Option 1: Use Master Notebook (Recommended for First Run)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/master_pipeline.ipynb
# Run all cells to:
# - Load and preprocess resume_data.csv
# - Train all ML models
# - Generate visualizations
# - Save trained models
```

**Time to complete:** 5-10 minutes

---

### Option 2: Launch Web Interface

#### Terminal 1: Streamlit App
```bash
# Start web interface
streamlit run app.py

# Opens: http://localhost:8501
```

#### Terminal 2: FastAPI Backend (in new terminal)
```bash
# Start API server
python api.py

# Available at: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## 📊 First-Time Usage

### 1. Resume Analysis
- Go to "👤 Resume Analysis" tab in Streamlit
- Paste or upload a resume
- View extracted skills, education, experience

### 2. Resume-Job Matching
- Go to "🎯 Job Matching" tab
- Paste candidate resume
- Paste job description
- Click "Calculate Match Score"
- View detailed breakdown

### 3. ML Predictions
- Use FastAPI endpoint for predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [8.5, 5, 3, 2, 2]}'
```

---

## 🔑 Key Endpoints

### Resume Analysis
```
POST /extract-skills
Input: text
Output: skills, education, experience
```

### Matching Engine
```
POST /match-score
Input: resume_text, job_description
Output: match_score (0-100), breakdown
```

### ML Prediction
```
POST /predict
Input: features array, model_name
Output: prediction, confidence score
```

### LLM Features (requires API key)
```
POST /resume-summary → Generate professional summary
POST /interview-questions → Generate 5 interview questions
POST /hiring-feedback → Generate hiring decision feedback
```

### Chatbot (RAG-based)
```
POST /chat
Input: message (recruiter question)
Output: chatbot response with context
```

---

## 📈 Project Structure Overview

```
├── resume_data.csv                 ← Your dataset
├── app.py                          ← Streamlit web interface
├── api.py                          ← FastAPI backend
├── config/settings.py              ← Configuration
├── src/
│   ├── data_preprocessing.py       ← Data cleaning
│   ├── nlp_processor.py            ← NLP & skills
│   ├── matching_engine.py          ← Resume-job matching
│   ├── ml_models.py                ← ML training
│   ├── feature_engineering.py      ← Advanced features
│   ├── model_persistence.py        ← Save/load models
│   └── llm_integration.py          ← LLM & chatbot
├── models/                         ← Saved trained models
└── notebooks/
    └── master_pipeline.ipynb       ← Complete pipeline
```

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] NLTK downloaded successfully
- [ ] API starts without errors: `python api.py`
- [ ] Streamlit launches: `streamlit run app.py`
- [ ] Can upload resume in web app
- [ ] Can enter job description
- [ ] Match score calculates (shows 0-100)
- [ ] Skills are extracted from text
- [ ] Matches appear in dashboard

---

## 🐛 Common Issues

### Issue: "NLTK data not found"
```bash
python -m nltk.downloader punkt stopwords wordnet
```

### Issue: "Port 8000 already in use"
```bash
# Use different port for FastAPI
python -c "
import uvicorn
from api import app
uvicorn.run(app, host='0.0.0.0', port=8001)
"
```

### Issue: "OpenAI API key not configured"
- LLM features (summary, questions, feedback) won't work
- Other features (matching, predictions) work fine
- Add API key to .env file to enable

### Issue: "Out of memory"
- Reduce VECTORIZER_MAX_FEATURES in config/settings.py
- Use subset of data for testing
- Process files in batches

---

## 🎓 Learning Resources

### Inside This Project
1. **Notebook**: `notebooks/master_pipeline.ipynb` - See full pipeline
2. **Source Code**: `src/*.py` - Study implementation
3. **API Docs**: http://localhost:8000/docs - Interactive API explorer
4. **README.md** - Comprehensive documentation

### External Resources
- **ML**: https://scikit-learn.org/stable/
- **NLP**: https://www.nltk.org/
- **Streamlit**: https://docs.streamlit.io/
- **FastAPI**: https://fastapi.tiangolo.com/

---

## 🚀 Next Steps

1. **Load Your Data**
   - Replace `resume_data.csv` with your dataset
   - Follow same format (columns: candidate, job, skills, etc.)

2. **Train Models**
   - Run master notebook with your data
   - Models trained: Random Forest, SVM, Logistic Regression, Decision Tree

3. **Deploy**
   - Configure .env with your API keys
   - Deploy Streamlit to Streamlit Cloud
   - Deploy FastAPI to Heroku/AWS/GCP

4. **Customize**
   - Adjust weights in matching engine
   - Add custom skills to NLP processor
   - Modify clustering parameters

---

## 📞 Quick Help

```bash
# Test if setup is correct
python -c "
import pandas as pd
import nltk
from sklearn.ensemble import RandomForestClassifier
print('✅ All core libraries working!')
"

# Check models directory
ls -la models/

# View recent models
python -c "
from src.model_persistence import ModelPersistence
mp = ModelPersistence()
print(mp.list_models())
"
```

---

## 🎉 Success!

You're ready to:
- ✅ Analyze resumes and extract skills
- ✅ Match candidates to jobs (0-100 score)
- ✅ Predict candidate selection with ML
- ✅ Generate summaries and interview questions (with LLM)
- ✅ Use intelligent recruiting chatbot

**Start with:** `jupyter notebook notebooks/master_pipeline.ipynb`

---

## 📝 Notes

- All features work without API keys (except LLM features)
- Models saved in `models/` directory
- Results cached for faster retrieval
- Logs available for debugging
- Configuration in `config/settings.py`

---

**Let's build better hiring! 🚀**
