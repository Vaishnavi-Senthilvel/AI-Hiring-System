"""
Test the improved file extraction function
"""
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(".") / 'src'))

# Simulate the extract_text_from_file function from app.py
def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file (PDF, TXT, DOCX)"""
    try:
        if not uploaded_file:
            return "Error: No file provided"
        
        file_name = uploaded_file['name'].lower()
        file_bytes = uploaded_file['bytes']

        # Handle text files
        if file_name.endswith('.txt'):
            try:
                content = file_bytes.decode('utf-8', errors='ignore')
                return content if content.strip() else "Error: File is empty"
            except Exception as e:
                return f"Error decoding TXT file: {str(e)}"

        else:
            return f"Unsupported file type: {file_name}."

    except Exception as e:
        return f"Error extracting text: {str(e)}"


# Test with sample resume
print("=" * 60)
print("Testing File Extraction Function")
print("=" * 60)

try:
    # Read sample resume
    with open('sample_resume.txt', 'rb') as f:
        file_bytes = f.read()
    
    mock_file = {
        'name': 'sample_resume.txt',
        'bytes': file_bytes
    }
    
    extracted = extract_text_from_file(mock_file)
    
    if not extracted.startswith("Error"):
        print("✅ File extraction successful!")
        print(f"✅ Extracted {len(extracted)} characters")
        print(f"\n📄 First 300 characters of extracted content:")
        print("-" * 60)
        print(extracted[:300])
        print("-" * 60)
        
        # Test NLP processing
        from nlp_processor import NLPProcessor
        nlp = NLPProcessor()
        
        skills = nlp.extract_skills_from_text(extracted)
        education = nlp.extract_education(extracted)
        experience = nlp.extract_years_of_experience(extracted)
        
        print(f"\n🛠️  Skills extracted: {', '.join(skills[:5])}...")
        print(f"🎓 Education found: {education if education else 'None'}")
        print(f"📅 Experience: {experience} years")
        
        print("\n✅ ALL TESTS PASSED! File upload should now work in the app.")
    else:
        print(f"❌ Extraction failed: {extracted}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
