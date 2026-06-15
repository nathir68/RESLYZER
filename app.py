import fitz  # PyMuPDF
import re
import os
import io
import traceback
import spacy
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import docx  # python-docx
from PIL import Image
import pytesseract
import webbrowser
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MACHINE LEARNING SETUP ---

print("Loading NLP Model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define Job Descriptions (The "Training Data" for ML)
JOB_DESCRIPTIONS = {
    "Python Developer": "Python programming, Django, Flask, SQL, REST API, backend engineering, debugging, unit testing, Git, software development life cycle.",
    "Data Scientist": "Python, Machine Learning, Data Analysis, Statistics, Pandas, NumPy, Scikit-learn, TensorFlow, Deep Learning, SQL, Data Visualization, Matplotlib.",
    "Frontend Developer": "JavaScript, React, HTML, CSS, Angular, Vue, Web development, UI/UX, responsive design, frontend frameworks, DOM manipulation, Bootstrap, Tailwind.",
    "Project Manager": "Leadership, Agile, Scrum, Project Management, Communication, Planning, Risk Management, Teamwork, Jira, Trello, stakeholder management, budgeting.",
    "DevOps Engineer": "AWS, Docker, Kubernetes, CI/CD, Linux, Terraform, Cloud computing, Jenkins, Automation, Scripting, Azure, Google Cloud Platform.",
    "Java Developer": "Java, Spring Boot, Hibernate, JVM, OOP, Microservices, SQL, Multithreading, Enterprise application development, JUnit, Maven, Gradle.",
    "Mobile Developer": "Android, iOS, Flutter, React Native, Swift, Kotlin, Mobile application development, UI design, API integration, Google Play Store, App Store.",
}

# Define Keywords for the "Details" page UI (so "X of Y skills" still works)
SKILL_KEYWORDS = [
    'Python', 'Java', 'C++', 'JavaScript', 'React', 'Angular', 'Vue',
    'Node.js', 'Django', 'Flask', 'Spring', 'Ruby on Rails',
    'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'NoSQL',
    'Data Analysis', 'Data Science', 'Machine Learning', 'Deep Learning',
    'TensorFlow', 'PyTorch', 'scikit-learn', 'Pandas', 'NumPy',
    'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes',
    'Git', 'CI/CD', 'Agile', 'Scrum', 'Project Management',
    'HTML', 'CSS', 'TypeScript', 'Linux', 'Communication', 'Leadership'
]

# Helper map for UI details
JOB_REQUIREMENTS = {job: set(desc.lower().replace(',', '').split()) for job, desc in JOB_DESCRIPTIONS.items()}

# --- ML Helper Functions ---

def clean_text(text):
    """NLP Preprocessing"""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def get_ml_recommendations(resume_text):
    """
    Calculates Cosine Similarity + Keyword Stats for UI.
    Returns list of dicts with 'job', 'score', 'matching_count', 'total_count'
    """
    # 1. Clean Resume
    cleaned_resume = clean_text(resume_text)
    
    # 2. Vectorize & Calculate Similarity
    job_titles = list(JOB_DESCRIPTIONS.keys())
    job_texts = [clean_text(desc) for desc in JOB_DESCRIPTIONS.values()]
    corpus = [cleaned_resume] + job_texts
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # 3. Keyword Extraction (For the UI "X of Y skills" text)
    resume_skills = set()
    for skill in SKILL_KEYWORDS:
        if re.search(r"\b" + re.escape(skill) + r"\b", resume_text, re.IGNORECASE):
            resume_skills.add(skill)

    # 4. Format Results
    recommendations = []
    for i, score in enumerate(cosine_sim):
        job_name = job_titles[i]
        
        # Calculate simple counts for UI display
        req_skills = [s for s in SKILL_KEYWORDS if s.lower() in JOB_DESCRIPTIONS[job_name].lower()]
        match_count = len([s for s in req_skills if s in resume_skills])
        
        tfidf_score = score * 100
        keyword_score = (match_count / len(req_skills)) * 100 if len(req_skills) > 0 else 0
        
        # Hybrid Score: 75% weighted to exact skill matches, 25% to overall TF-IDF contextual similarity
        hybrid_score = round((keyword_score * 0.75) + (tfidf_score * 0.25), 1)
        
        if hybrid_score > 10: # Only show relevant jobs
            recommendations.append({
                'name': job_name,
                'pct': hybrid_score,       # The new Hybrid score
                'matching': match_count,   # For UI text
                'total': len(req_skills)   # For UI text
            })
    
    # Sort by ML Score
    recommendations.sort(key=lambda x: x['pct'], reverse=True)
    return recommendations, list(resume_skills)

# --- File Processing ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath, ext):
    if ext == 'pdf':
        try:
            doc = fitz.open(filepath)
            text = "".join([page.get_text() for page in doc])
            doc.close()
            return text
        except: return "" 
    elif ext == 'docx':
        try:
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        except: return ""
    return ""

# --- API Endpoints ---

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'resumeFile' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['resumeFile']
    if file.filename == '': return jsonify({'error': 'No file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        text = extract_text(filepath, ext)
        
        # Call the new ML Function
        jobs, skills_found = get_ml_recommendations(text)

        # Structure matches your frontend expectation
        return jsonify({
            'success': True,
            'skills_found': sorted(skills_found),
            'recommendations': jobs # Contains ML score inside 'pct'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/job_details/<path:job_name>', methods=['GET'])
def get_job_details(job_name):
    # Retrieve description and simple keywords for the Details page
    desc = JOB_DESCRIPTIONS.get(job_name, "")
    req_skills = [s for s in SKILL_KEYWORDS if s.lower() in desc.lower()]
    return jsonify({
        'job_title': job_name,
        'required_skills': sorted(list(set(req_skills)))
    })

# --- Routes for Serving HTML ---
@app.route('/')
def route_index(): 
    return render_template('index.html')

@app.route('/results')
def route_results(): 
    return render_template('results.html')

@app.route('/details')
def route_details(): 
    return render_template('details.html')

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new_tab('http://127.0.0.1:5000/')
    
    print("Starting ML Resume Intelligence...")
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True, port=5000, use_reloader=False)