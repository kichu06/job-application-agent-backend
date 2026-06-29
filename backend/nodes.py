from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Primary LLM — Groq
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_TOKEN"),
    temperature=0.3
)

# Fallback LLM — Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_TOKEN"),
    temperature=0.3
)

def invoke_with_fallback(chain_prompt, inputs):
    """Try Groq first, fall back to Gemini if it fails"""
    try:
        chain = chain_prompt | groq_llm
        return chain.invoke(inputs)
    except Exception as e:
        print(f"Groq failed: {e} — switching to Gemini")
        chain = chain_prompt | gemini_llm
        return chain.invoke(inputs)

def clean_json_response(text):
    """Remove markdown code blocks if LLM wraps JSON in them"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ─────────────────────────────────────────
# NODE 1 — Extract skills from job description
# ─────────────────────────────────────────
def extract_jd_skills(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a job description analyzer. 
         Extract information and return ONLY valid JSON, no extra text."""),
        ("human", """Analyze this job description and return JSON with this exact structure:
         {{
           "required_skills": ["extract ALL required skills mentioned, as many as exist"],
           "optional_skills": ["extract ALL optional/nice-to-have skills, as many as exist"],
           "seniority": "extract exact seniority level from JD as a string",
           "role_type": "extract exact role type from JD as a string",
           "company_name": "extract company name if mentioned, else Unknown",
           "job_title": "extract exact job title from JD"
         }}
         
         Extract everything exactly as mentioned in the job description.
         Do not limit or predefined any values — use whatever is in the JD.
         
         Job Description:
         {jd_text}""")
    ])

    response = invoke_with_fallback(prompt, {"jd_text": state["jd_text"]})
    extracted = json.loads(clean_json_response(response.content))
    return {"extracted_skills": extracted}


# ─────────────────────────────────────────
# NODE 2 — Parse resume text into structured data
# ─────────────────────────────────────────
def parse_resume(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a resume parser.
         Extract information and return ONLY valid JSON, no extra text."""),
        ("human", """Parse this resume and return JSON with this exact structure:
         {{
           "candidate_name": "full name",
           "your_skills": ["extract ALL skills mentioned anywhere in resume"],
           "experience_years": 0,
           "education": "degree and field",
           "recent_role": "most recent job title",
           "projects": ["project name: one line description"]
         }}
         
         Extract only actual technology names and tools.
         Do NOT include section headings as skills.

         Resume:
         {resume_text}""")
    ])

    response = invoke_with_fallback(prompt, {"resume_text": state["resume_text"]})
    parsed = json.loads(clean_json_response(response.content))
    return {"parsed_resume": parsed}


# ─────────────────────────────────────────
# NODE 3 — Gap analysis
# ─────────────────────────────────────────
def analyze_gaps(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a career advisor.
         Compare skills and return ONLY valid JSON, no extra text."""),
        ("human", """You are comparing what a job REQUIRES vs what a candidate HAS.
         
        TASK:
         - matched_skills = skills from REQUIRED list that candidate also has
         - missing_skills = skills from REQUIRED list that candidate does NOT have
         - Do NOT list candidate's extra skills as missing
         - match_score = percentage of required skills the candidate has

         Return JSON with this exact structure:
         {{
           "matched_skills": ["ALL skills that match"],
           "missing_skills": ["ALL skills that are missing"],
           "match_score": "a percentage between 0 and 100"
         }}
         
         Required skills: {required_skills}
         Candidate skills: {your_skills}""")
    ])

    response = invoke_with_fallback(prompt, {
        "required_skills": state["extracted_skills"]["required_skills"],
        "your_skills": state["parsed_resume"]["your_skills"]
    })
    gaps = json.loads(clean_json_response(response.content))
    return {"skill_gaps": gaps}


# ─────────────────────────────────────────
# NODE 4 — Generate cover letter
# ─────────────────────────────────────────
def generate_cover_letter(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert cover letter writer."),
        ("human", """Write a professional cover letter for this candidate.
         
         Company: {company_name}
         Job Title: {job_title}
         Role applying for: {role_type}
         Candidate name: {candidate_name}
         Their experience: {experience_years} years
         Their recent role: {recent_role}
         Their projects: {projects}
         Matched skills: {matched_skills}
         Missing skills: {missing_skills}
         
         Write a confident, tailored cover letter. 3 paragraphs max.""")
    ])

    response = invoke_with_fallback(prompt, {
        "company_name": state["extracted_skills"]["company_name"],
        "job_title": state["extracted_skills"]["job_title"],
        "role_type": state["extracted_skills"]["role_type"],
        "candidate_name": state["parsed_resume"]["candidate_name"],
        "experience_years": state["parsed_resume"]["experience_years"],
        "recent_role": state["parsed_resume"]["recent_role"],
        "projects": state["parsed_resume"]["projects"],
        "matched_skills": state["skill_gaps"]["matched_skills"],
        "missing_skills": state["skill_gaps"]["missing_skills"]
    })
    return {"cover_letter": response.content}


# ─────────────────────────────────────────
# NODE 5 — Generate interview questions
# ─────────────────────────────────────────
def generate_interview_questions(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an interview coach."),
        ("human", """Generate 5 likely interview questions with suggested answers.
         
         Job Title: {job_title}
         Role Type: {role_type}
         Candidate matched skills: {matched_skills}
         Candidate skill gaps: {missing_skills}
         
         Return ONLY valid JSON with this structure:
         {{
           "questions": [
             {{
               "question": "question text",
               "suggested_answer": "answer text"
             }}
           ]
         }}""")
    ])

    response = invoke_with_fallback(prompt, {
        "job_title": state["extracted_skills"]["job_title"],
        "role_type": state["extracted_skills"]["role_type"],
        "matched_skills": state["skill_gaps"]["matched_skills"],
        "missing_skills": state["skill_gaps"]["missing_skills"]
    })
    questions = json.loads(clean_json_response(response.content))
    return {"interview_questions": questions["questions"]}  