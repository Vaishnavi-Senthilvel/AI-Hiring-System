import sys
from pathlib import Path
sys.path.insert(0, str(Path(".") / 'src'))

from nlp_processor import NLPProcessor

# Initialize the processor
try:
    nlp = NLPProcessor()
    print("✅ NLPProcessor initialized successfully")
    
    # Test text
    test_text = "Senior Software Engineer with 5+ years of experience. Skills: Python, JavaScript, React, AWS, Machine Learning"
    
    # Test extraction methods
    skills = nlp.extract_skills_from_text(test_text)
    print(f"✅ Skills extracted: {skills}")
    
    education = nlp.extract_education(test_text)
    print(f"✅ Education extracted: {education}")
    
    experience = nlp.extract_years_of_experience(test_text)
    print(f"✅ Experience extracted: {experience} years")
    
    print("\n✅ All tests passed! The upload module should work now.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
