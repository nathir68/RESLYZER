// --- Global Configuration ---
const FLASK_API_URL = '/recommend';

// --- Utility Functions ---
function displayError(message, containerId, messageId) {
    const errorContainer = document.getElementById(containerId);
    const errorMessageElement = document.getElementById(messageId);
    if (errorContainer && errorMessageElement) {
        errorContainer.classList.remove('hidden');
        errorMessageElement.textContent = message;
    }
}

function hideError(containerId) {
    const errorContainer = document.getElementById(containerId);
    if (errorContainer) {
        errorContainer.classList.add('hidden');
    }
}

// --- Icons (SVG Strings) ---
const iconCheck = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="badge-icon"><path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" /></svg>`;
const iconX = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="badge-icon"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" /></svg>`;
const iconArrow = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="arrow-icon"><path stroke-linecap="round" stroke-linejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" /></svg>`;

// --- Index Page Logic ---
function initUploadPage() {
    const fileInput = document.getElementById('resumeFile');
    const analyzeButton = document.getElementById('analyzeButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const uploadZone = document.getElementById('uploadZone');

    if (!fileInput) return;

    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function() {
        if (this.files[0]) {
            fileName.textContent = this.files[0].name;
            fileInfo.classList.remove('hidden');
        } else {
            fileInfo.classList.add('hidden');
        }
    });

    analyzeButton.addEventListener('click', async function() {
        const file = fileInput.files[0];
        if (!file) {
            displayError("Please select a PDF or DOCX file to analyze.", 'errorContainer', 'errorMessage');
            return;
        }
        
        const formData = new FormData();
        formData.append('resumeFile', file);
        
        hideError('errorContainer');
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
            displayError(`${error.message}. Ensure Flask server is running.`, 'errorContainer', 'errorMessage');
        } finally {
            loadingIndicator.classList.add('hidden');
            analyzeButton.disabled = false;
        }
    });
}

// --- Results Page Logic ---
function displaySkills(skills, listElement) {
    listElement.innerHTML = '';
    if (!skills || skills.length === 0) {
        listElement.innerHTML = '<p style="color: var(--color-text-secondary); font-size: 0.9rem;">No skills detected</p>';
        return;
    }
    skills.forEach(skill => {
        const span = document.createElement('span');
        span.className = 'skill-badge';
        span.innerHTML = `${iconCheck}${skill}`;
        listElement.appendChild(span);
    });
}

function getScoreClass(percentage) {
    if (percentage >= 80) return 'score-high';
    if (percentage >= 60) return 'score-medium';
    if (percentage >= 40) return 'score-low';
    return 'score-poor';
}

function displayRankedJobs(jobScores, listElement) {
    listElement.innerHTML = '';
    if (!jobScores || jobScores.length === 0) {
        listElement.innerHTML = '<p style="color: var(--color-text-secondary); font-size: 0.9rem; text-align: center;">No recommendations found</p>';
        return;
    }
    
    jobScores.forEach((jobData) => {
        const a = document.createElement('a');
        a.href = `/details?job=${encodeURIComponent(jobData.name)}`;
        a.className = 'job-card';
        
        const percentage = jobData.pct.toFixed(0);
        const scoreClass = getScoreClass(percentage);

        a.innerHTML = `
            <div class="job-score ${scoreClass}">
                ${percentage}%
            </div>
            <div class="job-info">
                <h3>${jobData.name}</h3>
                <p>ML Score: ${percentage}% &bull; Matches ${jobData.matching} of ${jobData.total} key skills</p>
            </div>
            ${iconArrow}
        `;
        listElement.appendChild(a);
    });
}

function initResultsPage() {
    const skillsListElement = document.getElementById('skillsFoundList');
    const recommendationsListElement = document.getElementById('recommendationsList');
    if (!skillsListElement || !recommendationsListElement) return;

    const resultsData = localStorage.getItem('analysisResults');
    if (!resultsData) {
        window.location.href = '/';
        return;
    }

    const results = JSON.parse(resultsData);
    displaySkills(results.skills_found, skillsListElement);
    displayRankedJobs(results.recommendations, recommendationsListElement);
}

// --- Details Page Logic ---
function displayList(items, listElement, type = 'default') {
    listElement.innerHTML = '';
    if (!items || items.length === 0) {
        listElement.innerHTML = `<span class="skill-badge" style="opacity: 0.5;">None</span>`;
        return;
    }
    
    items.forEach(item => {
        const span = document.createElement('span');
        let icon = '';
        
        if (type === 'match') {
            span.className = `skill-badge success`;
            icon = iconCheck;
        } else if (type === 'miss') {
            span.className = `skill-badge danger`;
            icon = iconX;
        } else {
            span.className = `skill-badge`;
        }
        
        span.innerHTML = `${icon}${item}`;
        listElement.appendChild(span);
    });
}

async function initDetailsPage() {
    const jobTitleElement = document.getElementById('jobTitle');
    const progressFillElement = document.getElementById('progressFill');
    const matchingSkillsList = document.getElementById('matchingSkillsList');
    const missingSkillsList = document.getElementById('missingSkillsList');
    
    if (!jobTitleElement) return;

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
        const jobRec = JSON.parse(resultsData).recommendations.find(r => r.name === jobName);
        if (jobRec) {
            percentage = jobRec.pct;
        } else if (requiredSkills.size > 0) {
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
