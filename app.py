import fitz  # PyMuPDF
import re
import os
import io
import traceback
import spacy
import numpy as np
from flask import Flask, request, jsonify
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
        
        match_pct = round(score * 100, 1) # The smart ML score
        
        if match_pct > 10: # Only show relevant jobs
            recommendations.append({
                'name': job_name,
                'pct': match_pct,          # ML Score
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

# --- FRONTEND (EXACTLY YOUR UI CODE) ---

HTML_HEAD = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .upload-zone { transition: all 0.3s ease; border: 2px dashed #cbd5e1; }
        .upload-zone:hover { border-color: #667eea; background: #f8fafc; }
        .skill-badge { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .job-card { transition: all 0.2s ease; text-decoration: none; }
        .job-card:hover { transform: translateX(5px); }
        .progress-bar { background-color: #e5e7eb; border-radius: 9999px; overflow: hidden; height: 1.5rem; }
        .progress-fill {
            background-color: #22c55e; height: 100%;
            transition: width 0.5s ease-out; text-align: center;
            font-weight: 600; color: white; line-height: 1.5rem;
        }
        .loader-dots div {
            width: 0.75rem; height: 0.75rem;
            background-color: #667eea; border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .loader-dots .dot-1 { animation-delay: -0.32s; }
        .loader-dots .dot-2 { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.0); }
        }
    </style>
</head>
"""

HTML_INDEX = """
<!DOCTYPE html>
<html lang="en">
""" + HTML_HEAD + """
<body class="p-6">
    <div class="max-w-5xl mx-auto">
        <div class="text-center mb-8">
            <div class="inline-flex items-center justify-center w-16 h-16 bg-white rounded-2xl shadow-lg mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-8 h-8 text-purple-600">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                </svg>
            </div>
            <h1 class="text-4xl font-bold text-white mb-2">Resume Intelligence</h1>
            <p class="text-purple-100 text-lg">AI-powered career path analysis</p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-1">
                <div class="glass-card rounded-2xl p-6 shadow-xl">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-2 text-purple-600">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                        </svg>
                        Upload Resume
                    </h2>
                    
                    <div class="upload-zone rounded-xl p-8 text-center mb-4 cursor-pointer" onclick="document.getElementById('resumeFile').click()">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-12 h-12 mx-auto text-gray-400 mb-3">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                        </svg>
                        <p class="text-sm text-gray-600 mb-1">Click to browse</p>
                        <p class="text-xs text-gray-400">PDF or DOCX files</p>
                    </div>
                    
                    <input type="file" id="resumeFile" accept=".pdf,.docx" class="hidden"/>
                    
                    <div id="fileInfo" class="hidden mb-4 p-3 bg-purple-50 rounded-lg">
                        <p class="text-sm text-purple-800 font-medium" id="fileName"></p>
                    </div>
                    
                    <button id="analyzeButton" onclick="handleAnalysis()"
                            class="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-indigo-700 transition duration-200 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed">
                        Analyze Resume
                    </button>
                    
                    <div id="loadingIndicator" class="hidden mt-4 text-center">
                        <div class="flex items-center justify-center space-x-1 mb-2">
                            <div class="w-2 h-2 rounded-full bg-purple-600 animate-bounce"></div>
                            <div class="w-2 h-2 rounded-full bg-purple-600 animate-bounce" style="animation-delay: 0.1s"></div>
                            <div class="w-2 h-2 rounded-full bg-purple-600 animate-bounce" style="animation-delay: 0.2s"></div>
                        </div>
                        <p class="text-sm text-gray-600">Processing with Machine Learning...</p>
                    </div>
                    
                    <div id="errorContainer" class="hidden mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
                        <p class="text-sm font-semibold text-red-800 mb-1">Analysis Failed</p>
                        <p class="text-xs text-red-600" id="errorMessage"></p>
                    </div>
                </div>
            </div>

            <div class="lg:col-span-2">
                <div id="emptyState" class="glass-card rounded-2xl p-12 shadow-xl text-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="w-20 h-20 mx-auto text-gray-300 mb-4">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25ZM6.75 12h.008v.008H6.75V12Zm0 3h.008v.008H6.75V15Zm0 3h.008v.008H6.75V18Z" />
                    </svg>
                    <h3 class="text-xl font-semibold text-gray-400 mb-2">No Analysis Yet</h3>
                    <p class="text-gray-400">Upload your resume to see your skill analysis and job recommendations here.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const FLASK_API_URL = '/recommend';
        const fileInput = document.getElementById('resumeFile');
        const analyzeButton = document.getElementById('analyzeButton');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const errorContainer = document.getElementById('errorContainer');
        const errorMessageElement = document.getElementById('errorMessage');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');

        fileInput.addEventListener('change', function() {
            if (this.files[0]) {
                fileName.textContent = this.files[0].name;
                fileInfo.classList.remove('hidden');
            } else {
                fileInfo.classList.add('hidden');
            }
        });

        function displayError(message) {
            errorContainer.classList.remove('hidden');
            errorMessageElement.textContent = message;
        }

        async function handleAnalysis() {
            const file = fileInput.files[0];
            if (!file) {
                displayError("Please select a PDF or DOCX file to analyze.");
                return;
            }
            const formData = new FormData();
            formData.append('resumeFile', file);
            
            errorContainer.classList.add('hidden');
            loadingIndicator.classList.remove('hidden');
            analyzeButton.disabled = true;

            try {
                const response = await fetch(FLASK_API_URL, {
                    method: 'POST',
                    body: formData,
                });
                const analysis = await response.json();
                if (!response.ok || analysis.error) {
                    throw new Error(analysis.error || `HTTP Error ${response.status}`);
                }
                
                localStorage.setItem('analysisResults', JSON.stringify(analysis));
                window.location.href = '/results';

            } catch (error) {
                console.error('Full Error:', error);
                displayError(`${error.message}. Ensure Flask server is running.`);
            } finally {
                loadingIndicator.classList.add('hidden');
                analyzeButton.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

HTML_RESULTS = """
<!DOCTYPE html>
<html lang="en">
""" + HTML_HEAD + """
<body class="p-6">
    <div class="max-w-5xl mx-auto">
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold text-white mb-2">Your Analysis Results</h1>
            <a href="/" class="text-purple-100 text-lg hover:underline">&larr; Upload Another Resume</a>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="glass-card rounded-2xl p-6 shadow-xl">
                <h2 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-2 text-purple-600">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
                    </svg>
                    Skills Detected
                </h2>
                <div id="skillsFoundList" class="flex flex-wrap gap-2">
                    </div>
            </div>

            <div class="glass-card rounded-2xl p-6 shadow-xl">
                <h2 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-2 text-purple-600">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M20.25 14.15v4.25c0 1.094-.787 2.036-1.872 2.18-2.087.277-4.216.42-6.378.42s-4.291-.143-6.378-.42c-1.085-.144-1.872-1.086-1.872-2.18v-4.25m16.5 0a2.18 2.18 0 0 0 .75-1.661V8.706c0-1.081-.768-2.015-1.837-2.175a48.114 48.114 0 0 0-3.413-.387m4.5 8.006c-.194.165-.42.295-.673.38A23.978 23.978 0 0 1 12 15.75c-2.648 0-5.195-.429-7.577-1.22a2.016 2.016 0 0 1-.673-.38m0 0A2.18 2.18 0 0 1 3 12.489V8.706c0-1.081.768-2.015 1.837-2.175a48.111 48.111 0 0 1 3.413-.387m7.5 0V5.25A2.25 2.25 0 0 0 13.5 3h-3a2.25 2.25 0 0 0-2.25 2.25v.894m7.5 0a48.667 48.667 0 0 0-7.5 0M12 12.75h.008v.008H12v-.008Z" />
                    </svg>
                    Ranked Career Matches (ML Powered)
                </h2>
                <p class="text-sm text-gray-500 mb-4">Your recommended jobs, ranked by AI similarity.</p>
                <div id="recommendationsList" class="space-y-3">
                    <div id="jobLoading" class="flex flex-col items-center justify-center p-8">
                        <div class="loader-dots flex space-x-2">
                           <div class="dot-1"></div>
                           <div class="dot-2"></div>
                           <div class="dot-3"></div>
                        </div>
                        <p class="text-sm text-gray-600 mt-4">AI is calculating your match...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="/results.js"></script>
</body>
</html>
"""

HTML_DETAILS = """
<!DOCTYPE html>
<html lang="en">
""" + HTML_HEAD + """
<body class="p-6">
    <div class="max-w-5xl mx-auto">
        <div class="text-center mb-8">
            <h1 id="jobTitle" class="text-4xl font-bold text-white mb-2">Loading...</h1>
            <a href="/results" class="text-purple-100 text-lg hover:underline">&larr; Back to Recommendations</a>
        </div>

        <div class="glass-card rounded-2xl p-6 shadow-xl mb-6">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">
                Your Skill Match
            </h2>
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill" style="width: 0%;">0%</div>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="glass-card rounded-2xl p-6 shadow-xl">
                <h2 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-2 text-green-600">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                    </svg>
                    Your Matching Skills
                </h2>
                <div id="matchingSkillsList" class="flex flex-wrap gap-2">
                    </div>
            </div>

            <div class="glass-card rounded-2xl p-6 shadow-xl">
                <h2 class="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-2 text-red-600">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                    </svg>
                    Skills to Achieve 100%
                </h2>
                <div id="missingSkillsList" class="flex flex-wrap gap-2">
                    </div>
            </div>
        </div>
    </div>
    <script src="/details.js"></script>
</body>
</html>
"""

# --- Javascript Logic (Updated for ML but visually identical) ---

JS_RESULTS = """
function displaySkills(skills, listElement) {
    listElement.innerHTML = '';
    if (!skills || skills.length === 0) {
        listElement.innerHTML = '<p class="text-gray-400 text-sm">No skills detected</p>';
        return;
    }
    skills.forEach(skill => {
        const span = document.createElement('span');
        span.className = 'skill-badge inline-flex items-center px-3 py-1 bg-white border border-purple-200 text-purple-700 rounded-full text-sm font-medium shadow-sm';
        span.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-3 h-3 mr-1">
                <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
            ${skill}
        `;
        listElement.appendChild(span);
    });
}

function displayRankedJobs(jobScores, listElement) {
    listElement.innerHTML = '';
    if (!jobScores || jobScores.length === 0) {
        listElement.innerHTML = '<p class="text-gray-400 text-sm">No recommendations found</p>';
        return;
    }
    
    jobScores.forEach((jobData) => {
        const a = document.createElement('a');
        a.href = `/details?job=${encodeURIComponent(jobData.name)}`;
        a.className = 'job-card flex items-center p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl border border-purple-100 cursor-pointer';
        
        const percentage = jobData.pct.toFixed(0);
        let badgeColor = 'bg-red-500';
        if (percentage >= 80) badgeColor = 'bg-green-500';
        else if (percentage >= 60) badgeColor = 'bg-blue-500';
        else if (percentage >= 40) badgeColor = 'bg-yellow-500';
        else if (percentage >= 20) badgeColor = 'bg-orange-500';

        a.innerHTML = `
            <div class="flex-shrink-0 w-16 h-14 ${badgeColor} rounded-lg flex items-center justify-center text-white font-bold text-lg mr-4">
                ${percentage}%
            </div>
            <div>
                <p class="font-semibold text-gray-800">${jobData.name}</p>
                <p class="text-xs text-gray-500">ML Score: ${percentage}% | Matches ${jobData.matching} of ${jobData.total} key skills</p>
            </div>
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5 text-purple-400 ml-auto">
                <path stroke-linecap="round" stroke-linejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
            </svg>
        `;
        listElement.appendChild(a);
    });
}

function loadRankedJobs() {
    const skillsListElement = document.getElementById('skillsFoundList');
    const recommendationsListElement = document.getElementById('recommendationsList');

    const resultsData = localStorage.getItem('analysisResults');
    if (!resultsData) {
        window.location.href = '/';
        return;
    }

    const results = JSON.parse(resultsData);
    displaySkills(results.skills_found, skillsListElement);

    // The ML Backend already sorted and calculated these for us!
    // We just render them now.
    displayRankedJobs(results.recommendations, recommendationsListElement);
}

document.addEventListener('DOMContentLoaded', loadRankedJobs);
"""

JS_DETAILS = """
const jobTitleElement = document.getElementById('jobTitle');
const progressFillElement = document.getElementById('progressFill');
const matchingSkillsList = document.getElementById('matchingSkillsList');
const missingSkillsList = document.getElementById('missingSkillsList');

function displayList(items, listElement, type = 'default') {
    listElement.innerHTML = '';
    if (!items || items.length === 0) {
        listElement.innerHTML = `<span class="inline-flex items-center px-3 py-1 bg-gray-100 text-gray-500 rounded-full text-sm font-medium">None</span>`;
        return;
    }
    
    items.forEach(item => {
        const span = document.createElement('span');
        let icon = '';
        let baseClasses = 'skill-badge inline-flex items-center px-3 py-1 rounded-full text-sm font-medium shadow-sm';
        
        if (type === 'match') {
            span.className = `${baseClasses} bg-green-100 border border-green-200 text-green-800`;
            icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-3 h-3 mr-1"><path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" /></svg>`;
        } else if (type === 'miss') {
            span.className = `${baseClasses} bg-red-100 border border-red-200 text-red-800`;
            icon = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-3 h-3 mr-1"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" /></svg>`;
        }
        
        span.innerHTML = `${icon}${item}`;
        listElement.appendChild(span);
    });
}

async function loadJobDetails() {
    const params = new URLSearchParams(window.location.search);
    const jobName = params.get('job');
    if (!jobName) {
        jobTitleElement.textContent = "Error: No Job Selected";
        return;
    }
    jobTitleElement.textContent = jobName;

    const resultsData = localStorage.getItem('analysisResults');
    if (!resultsData) return;
    
    const userSkills = new Set(JSON.parse(resultsData).skills_found || []);

    try {
        const response = await fetch(`/job_details/${encodeURIComponent(jobName)}`);
        const data = await response.json();
        const requiredSkills = new Set(data.required_skills || []);

        const matchingSkills = [...requiredSkills].filter(skill => userSkills.has(skill));
        const missingSkills = [...requiredSkills].filter(skill => !userSkills.has(skill));

        let percentage = 0;
        if (requiredSkills.size > 0) {
            percentage = (matchingSkills.length / requiredSkills.size) * 100;
        } 
        
        const percentageText = percentage.toFixed(0) + '%';
        progressFillElement.style.width = percentageText;
        progressFillElement.textContent = percentageText;
        
        displayList(matchingSkills, matchingSkillsList, 'match');
        displayList(missingSkills, missingSkillsList, 'miss');
        
    } catch (error) {
        console.error("Failed to load details:", error);
    }
}
document.addEventListener('DOMContentLoaded', loadJobDetails);
"""

# --- Routes for Serving HTML ---
@app.route('/')
def route_index(): return HTML_INDEX

@app.route('/results')
def route_results(): return HTML_RESULTS

@app.route('/details')
def route_details(): return HTML_DETAILS

@app.route('/results.js')
def route_results_js(): return JS_RESULTS, 200, {'Content-Type': 'application/javascript'}

@app.route('/details.js')
def route_details_js(): return JS_DETAILS, 200, {'Content-Type': 'application/javascript'}

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new_tab('http://127.0.0.1:5000/')
    
    print("Starting ML Resume Intelligence...")
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True, port=5000, use_reloader=False)