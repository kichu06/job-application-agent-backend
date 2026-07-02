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
    temperature=0.0
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_TOKEN"),
    temperature=0.0
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


# ─────────────────────────────────────────
# OPTION C — Python deduplication (runs after Node 1)
# ─────────────────────────────────────────
def normalize_skill(skill: str) -> str:
    """Normalize skill for deduplication comparison."""
    # Remove version annotations: "JavaScript (ES6+)" → "javascript"
    skill = re.sub(r'\s*\(.*?\)', '', skill)
    # Remove punctuation
    skill = re.sub(r'[^\w\s]', '', skill)
    return skill.strip().lower()


def deduplicate_skills(skills: list) -> list:
    """Remove duplicate skills keeping the most specific version."""
    seen = {}
    for skill in skills:
        key = normalize_skill(skill)
        if not key:
            continue
        # Keep the longer (more specific) version
        if key not in seen or len(skill) > len(seen[key]):
            seen[key] = skill
    return list(seen.values())


# ─────────────────────────────────────────
# NODE 1 — Extract skills from job description
# Option B: structured extraction with categories separated
# ─────────────────────────────────────────
def extract_jd_skills(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a job description analyzer. 
         Extract information and return ONLY valid JSON, no extra text."""),
        ("human", """Analyze this job description carefully and return JSON 
         with this exact structure:
         {{
           "categories": {{
             "CategoryName": ["skill1", "skill2"]
           }},
           "required_skills": ["individual atomic skill names only"],
           "optional_skills": ["individual atomic skill names only"],
           "seniority": "extract exact seniority level from JD as a string",
           "role_type": "extract exact role type from JD as a string",
           "company_name": "extract company name if mentioned, else Unknown",
           "job_title": "extract exact job title from JD"
         }}

         CRITICAL RULES:
         - This job could be in ANY field — tech, marketing, sales, healthcare, 
           finance, design, education, or any other industry

         STEP 1 — Identify categories:
         - The JD may have section/category headers (e.g. "State Management", 
           "Build Tools", "Testing", "Frontend", "Version Control")
         - A category header is a label with multiple sub-items listed under it
         - Fill "categories" with these headers as keys and their items as values
         - This forces you to think about structure before extracting skills

         STEP 2 — Extract required_skills:
         - Extract ONLY the leaf items (actual skills), never the category headers
         - Each skill must be ONE specific atomic item: a tool, technology,
           certification, methodology, language, or named competency
         - If the same skill appears in multiple forms 
           (e.g. "JavaScript" and "JavaScript (ES6+)"),
           include ONLY the most specific version ("JavaScript (ES6+)")
         - DO NOT include category/section headers as skills
         - DO NOT include full sentences
         - Generic soft skills only if explicitly named, keep them short
         - Each item 1-5 words maximum

         STEP 3 — Extract optional_skills:
         - Same rules as required_skills
         - These are skills marked as "nice to have", "preferred", "bonus"

         Job Description:
         {jd_text}""")
    ])

    response = invoke_with_fallback(prompt, {"jd_text": state["jd_text"]})
    extracted = json.loads(clean_json_response(response.content))

    # Option C — Python deduplication after LLM extraction
    if "required_skills" in extracted:
        extracted["required_skills"] = deduplicate_skills(
            extracted["required_skills"]
        )
    if "optional_skills" in extracted:
        extracted["optional_skills"] = deduplicate_skills(
            extracted["optional_skills"]
        )

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
           "your_skills": ["individual atomic skill names only"],
           "experience_years": 0,
           "education": "degree and field",
           "recent_role": "most recent job title",
           "projects": ["project name: one line description"]
         }}

         CRITICAL RULES:
         - Extract ONLY skills that are EXPLICITLY written in the resume text below
         - DO NOT add, infer, or guess any skill not literally present in the text
         - DO NOT include section headings as skills
         - Each skill must be a single atomic item
         - Split comma or bullet separated items into individual entries
         - Double check: every item in your_skills must appear 
           word-for-word somewhere in the resume text

         Resume:
         {resume_text}""")
    ])
    response = invoke_with_fallback(prompt, {"resume_text": state["resume_text"]})
    parsed = json.loads(clean_json_response(response.content))

    # Also deduplicate resume skills
    if "your_skills" in parsed:
        parsed["your_skills"] = deduplicate_skills(parsed["your_skills"])

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
         - A skill can NEVER appear in both lists
         - Every skill in REQUIRED list must appear in exactly ONE list

         IMPORTANT — treat these as the SAME skill when matching:
         - "JavaScript" == "JavaScript (ES6+)" == "JS"
         - "React" == "React.js" == "ReactJS"  
         - "Node" == "Node.js" == "NodeJS"
         - "Postgres" == "PostgreSQL"
         - "TypeScript" == "TS"
         - Apply this logic for any technology with version suffixes, 
           dots, or common abbreviations

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

    # Python deduplication on matched/missing too
    gaps["matched_skills"] = deduplicate_skills(
        gaps.get("matched_skills", [])
    )
    gaps["missing_skills"] = deduplicate_skills(
        gaps.get("missing_skills", [])
    )

    # Remove any skill that appears in both lists (belt and suspenders)
    matched_normalized = {normalize_skill(s) for s in gaps["matched_skills"]}
    gaps["missing_skills"] = [
        s for s in gaps["missing_skills"]
        if normalize_skill(s) not in matched_normalized
    ]

    # Python-calculated match score
    matched_count = len(gaps.get("matched_skills", []))
    total_count = matched_count + len(gaps.get("missing_skills", []))
    score = round((matched_count / total_count) * 100) if total_count > 0 else 0
    gaps["match_score"] = str(score)

    return {"skill_gaps": gaps}


# ─────────────────────────────────────────
# NODE 4 — Compute ATS score (pure Python, no LLM)
# ─────────────────────────────────────────
def compute_ats_score(state):
    required_skills = state["extracted_skills"]["required_skills"]
    matched_skills = state["skill_gaps"]["matched_skills"]
    missing_skills = state["skill_gaps"]["missing_skills"]
    candidate_skills = state["parsed_resume"]["your_skills"]
    optional_skills = state["extracted_skills"].get("optional_skills", [])

    total_required = len(required_skills)
    matched_count = len(matched_skills)
    match_percentage = round(
        (matched_count / total_required) * 100
    ) if total_required else 100

    optional_matches = [
        skill for skill in optional_skills
        if normalize_skill(skill) in {normalize_skill(s) for s in candidate_skills}
    ]
    optional_bonus = min(10, len(optional_matches) * 2)

    required_normalized = {normalize_skill(s) for s in required_skills}
    optional_normalized = {normalize_skill(s) for s in optional_skills}
    extra_skills = [
        skill for skill in candidate_skills
        if normalize_skill(skill) not in required_normalized
        and normalize_skill(skill) not in optional_normalized
    ]
    extra_bonus = min(5, len(extra_skills) // 3)

    score = min(100, match_percentage + optional_bonus + extra_bonus)

    recommendations = []
    if missing_skills:
        recommendations.append(
            "Focus on the missing required skills: " + ", ".join(missing_skills)
        )
    if optional_skills:
        recommendations.append(
            "Strengthen the resume by highlighting optional skills such as: "
            + ", ".join(optional_skills)
        )
    elif extra_skills:
        recommendations.append(
            "Your resume includes additional skills beyond the JD requirements "
            "that can help ATS relevance: " + ", ".join(extra_skills[:5])
        )
    if not candidate_skills:
        recommendations.append(
            "Add more explicit skills to the resume to improve ATS matching."
        )

    return {
        "ats_score": {
            "score": score,
            "match_percentage": match_percentage,
            "matched_required_count": matched_count,
            "required_count": total_required,
            "optional_matches": optional_matches,
            "extra_skill_count": len(extra_skills),
            "missing_skills": missing_skills,
            "recommendations": recommendations
        }
    }


# ─────────────────────────────────────────
# NODE 5 — Generate cover letter
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


# ─────────────────────────────────────────
# NODE 6 — Generate interview questions
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


# ─────────────────────────────────────────
# NODE 7 — Translate output to German if requested
# ─────────────────────────────────────────
def translate_output(state):
    language = state.get("language", "en")

    if language != "de":
        return {}

    try:
        cover_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional translator specializing in 
             job application documents. Translate into formal German (Sie-form). 
             Return only the translated text, no explanation."""),
            ("human", """Translate the following cover letter into German.

             Keep technology names unchanged (React.js, Python, Docker, 
             FastAPI, Git, etc.).
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
            ("system", """You are a professional translator specializing in 
             job application documents. Translate into formal German (Sie-form). 
             Return only the translated text, no explanation."""),
            ("human", """Translate the following interview questions and 
             suggested answers into German.

             Keep technology names unchanged (React.js, Python, Docker, 
             FastAPI, Git, etc.).
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
                stripped = line.strip()
                if stripped.startswith("QUESTION:") or stripped.startswith("FRAGE:"):
                    question = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("ANSWER:") or stripped.startswith("ANTWORT:"):
                    answer = stripped.split(":", 1)[1].strip()
                elif answer:
                    answer += " " + stripped
                else:
                    question += " " + stripped

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