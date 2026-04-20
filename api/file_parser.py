import io
import re
from pypdf import PdfReader
from docx import Document

def clean_extracted_text(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    
    stop_phrases = [
        r'^(References|Bibliography|Works Cited)\s*$',
        r'^RECOMMENDED STORIES',
        r'^More in Opinion',
        r'^SUPPORT PHILSTAR',
        r'^About Us\s*\|\s*Contact Us'
    ]
    
    skip_exact = {
        "X", "OK", "Yes", "No", "Login", "SIGN IN", "^", "abtest", "abtest2", 
        "Trending", "Latest", "SIGN-UP", "or sign in with"
    }
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line) # preserve some spacing
            continue
            
        # Truncate at bibliographies or bottom-of-page articles headers
        if any(re.match(ph, stripped, re.IGNORECASE) for ph in stop_phrases):
            break
            
        # Skip standalone numbers
        if re.match(r'^\d+$', stripped):
            continue
            
        # Skip UI elements
        if stripped in skip_exact:
            continue
            
        if stripped.startswith("Copyright ©"):
            continue
            
        # Skip sharing buttons and ad markers
        nl = stripped.lower()
        if 'sharing button' in nl or '<#comments>' in nl or nl.startswith('advertisement'):
            continue
            
        # Skip purely navigational links like "* Philstar.com <url>" or "| Subscribe"
        if stripped.startswith('o ') or stripped.startswith('* ') or stripped.startswith('| '):
            href_stripped = re.sub(r'\<https?://[^\>]+\>', '', stripped).strip()
            if len(href_stripped) < 25: # mostly just menu items
                continue
                
        # Skip lines that are entirely URLs
        if re.match(r'^\<https?://.*\>$', stripped):
            continue
            
        # Skip lines that are mostly just a URL with very little accompanying text
        href_stripped = re.sub(r'\<https?://[^\>]+\>', '', stripped).strip()
        if '<http' in stripped and len(href_stripped) < 15:
            continue
            
        cleaned_lines.append(line)
        
    joined = '\n'.join(cleaned_lines)
    
    # Strip any remaining inline URLs
    joined = re.sub(r'\<https?://[^\>]+\>', '', joined)
    # Strip placeholder hash links
    joined = re.sub(r'\<\#.*?\>', '', joined)
    # Clean up citations like [1], [1, 2]
    joined = re.sub(r'\[\d+(?:[\s,\-]+\d+)*\]', '', joined)
    # Remove excessive blank lines
    joined = re.sub(r'\n{3,}', '\n\n', joined)
    
    return joined.strip()

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    text = ""
    
    if ext == 'txt':
        text = file_bytes.decode('utf-8', errors='ignore')
    elif ext == 'pdf':
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    elif ext == 'docx':
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT.")
        
    return clean_extracted_text(text)
