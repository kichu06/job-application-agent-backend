from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_TOKEN"),
    temperature=0.3
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_TOKEN"),
    temperature=0.3
)

def invoke_with_fallback(chain_prompt, inputs):
    try:
        chain = chain_prompt | groq_llm
        return chain.invoke(inputs)
    except Exception as e:
        print(f"Groq failed: {e} — switching to Gemini")
        chain = chain_prompt | gemini_llm
        return chain.invoke(inputs)


def clean_json_response(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text, flags=re.I)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return text.strip()


def extract_jd_skills(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a job description analyzer. 
         Extract information and return ONLY valid JSON, no extra text."""),
        ("human", """Analyze this job description and return JSON with this exact structure:
         {{
           "required_skills": ["individual atomic skill names only"],
           "optional_skills": ["individual atomic skill names only"],
           "seniority": "extract exact seniority level from JD as a string",
           "role_type": "extract exact role type from JD as a string",
           "company_name": "extract company name if mentioned, else Unknown",
           "job_title": "extract exact job title from JD"
         }}

         CRITICAL RULES FOR SKILLS EXTRACTION:
         - This job could be in ANY field — tech, marketing, sales, healthcare, 
           finance, design, education, or any other industry
         - Each skill must be ONE specific, atomic item — a tool, technology, 
           certification, methodology, language, or named competency
         - Break down compound bullet points into individual skills
         - DO NOT include full requirement sentences as a single skill
         - DO NOT copy JD bullet points verbatim
         - Generic soft skills should only be included if explicitly named,
           keep them short, e.g. "Communication Skills"
         - Each item should be 1-5 words maximum

         Job Description:
         {jd_text}""")
    ])
    
    response = invoke_with_fallback(prompt, {"jd_text": state["jd_text"]})
    extracted = json.loads(clean_json_response(response.content))
    return {"extracted_skills": extracted}


def parse_resume(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a resume parser.
         Extract information and return ONLY valid JSON, no extra text."""),
        ("human", """Parse this resume and return JSON with this exact structure:
         {{
           "candidate_name": "full name",
           "your_skills": ["individual atomic skill names only"],
           "experience_years": 0,
           "education": "degree and field",
           "recent_role": "most recent job title",
           "projects": ["project name: one line description"]
         }}

         CRITICAL RULES:
         - Extract ONLY skills that are EXPLICITLY written in the resume text below
         - DO NOT add, infer, or guess any skill that is not literally present
         - DO NOT include section headings as skills
         - Each skill must be a single atomic item
         - Split comma or bullet separated items into individual entries

         Resume:
         {resume_text}""")
    ])
    response = invoke_with_fallback(prompt, {"resume_text": state["resume_text"]})
    parsed = json.loads(clean_json_response(response.content))
    return {"parsed_resume": parsed}


def analyze_gaps(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a career advisor.
         Compare skills and return ONLY valid JSON, no extra text."""),
        ("human", """You are comparing what a job REQUIRES vs what a candidate HAS.

        TASK:
         - matched_skills = skills from REQUIRED list that candidate also has
         - missing_skills = skills from REQUIRED list that candidate does NOT have
         - A skill can appear in EITHER matched_skills OR missing_skills, NEVER both
         - Every skill in REQUIRED list must appear in exactly ONE of the two lists

         Return JSON with this exact structure:
         {{
           "matched_skills": ["skills present in both lists"],
           "missing_skills": ["required skills not in candidate list"]
         }}

         Required skills: {required_skills}
         Candidate skills: {your_skills}""")
    ])
    response = invoke_with_fallback(prompt, {
        "required_skills": state["extracted_skills"]["required_skills"],
        "your_skills": state["parsed_resume"]["your_skills"]
    })
    gaps = json.loads(clean_json_response(response.content))

    matched_count = len(gaps.get("matched_skills", []))
    total_count = matched_count + len(gaps.get("missing_skills", []))
    score = round((matched_count / total_count) * 100) if total_count > 0 else 0
    gaps["match_score"] = str(score)

    return {"skill_gaps": gaps}


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

         IMPORTANT RULES:
         - Do NOT use any placeholders like [Address], [Date], [Hiring Manager]
         - Start directly with "Dear Hiring Team," if manager name is unknown
         - Do not include address blocks
         - 3 paragraphs only
         - Write naturally using only the information provided above""")
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
             {{"question": "question text", "suggested_answer": "answer text"}}
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


def translate_output(state):
    language = state.get("language", "en")

    if language != "de":
        return {}

    try:
        cover_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional translator specializing in job application documents. Translate into formal German (Sie-form). Return only the translated text, no explanation."),
            ("human", """Translate the following cover letter into German.

Keep technology names unchanged (React.js, Python, Docker, FastAPI, Git, etc.).
Keep company names and proper nouns unchanged.
Preserve paragraph structure.

Cover letter:
{cover_letter}""")
        ])

        cover_response = invoke_with_fallback(cover_prompt, {
            "cover_letter": state["cover_letter"]
        })
        translated_cover_letter = cover_response.content.strip()

        questions_text = "\n\n".join(
            f"QUESTION: {q['question']}\nANSWER: {q['suggested_answer']}"
            for q in state.get("interview_questions", [])
        )

        questions_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional translator specializing in job application documents. Translate into formal German (Sie-form). Return only the translated text, no explanation."),
            ("human", """Translate the following interview questions and suggested answers into German.

Keep technology names unchanged (React.js, Python, Docker, FastAPI, Git, etc.).
Keep company names and proper nouns unchanged.
Preserve the question-answer structure.

{questions}""")
        ])

        questions_response = invoke_with_fallback(questions_prompt, {
            "questions": questions_text
        })
        translated_questions_text = questions_response.content.strip()

        translated_questions = []
        for block in translated_questions_text.split("\n\n"):
            if not block.strip():
                continue
            question = ""
            answer = ""
            for line in block.splitlines():
                if line.startswith("QUESTION:"):
                    question = line[len("QUESTION:"):].strip()
                elif line.startswith("ANSWER:"):
                    answer = line[len("ANSWER:"):].strip()
                elif answer:
                    answer += " " + line.strip()
                else:
                    question += " " + line.strip()

            if question or answer:
                translated_questions.append({
                    "question": question,
                    "suggested_answer": answer
                })

        return {
            "cover_letter": translated_cover_letter,
            "interview_questions": translated_questions
        }

    except Exception as e:
        print(f"Translation failed: {e} — falling back to English output")
        return {}
