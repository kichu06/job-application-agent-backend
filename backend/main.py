from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import io

from agent import agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"status": "AI Job Application Agent is running"}


@app.post("/run-agent")
async def run_agent(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    language: str = Form("en")
):
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    if not jd_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty"
        )

    try:
        pdf_bytes = await resume.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text() or ""

        if not resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. Make sure it is not scanned or image-based"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read PDF: {str(e)}"
        )

    try:
        result = agent.invoke({
            "jd_text": jd_text,
            "resume_text": resume_text,
            "language": language,
            "extracted_skills": None,
            "parsed_resume": None,
            "skill_gaps": None,
            "cover_letter": None,
            "interview_questions": None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )
    return {
        "extracted_skills": result["extracted_skills"],
        "parsed_resume": result["parsed_resume"],
        "skill_gaps": result["skill_gaps"],
        "ats_score": result.get("ats_score"),
        "cover_letter": result["cover_letter"],
        "interview_questions": result["interview_questions"]
    }

class RefineCoverLetterRequest(BaseModel):
    cover_letter: str
    instruction: str
    context: dict

@app.post("/refine-cover-letter")
async def refine_cover_letter(request: RefineCoverLetterRequest):
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from nodes import invoke_with_fallback

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert cover letter writer. Refine the cover letter based on the instruction. Return only the updated cover letter text, no explanation, no preamble."),
            ("human", """Refine this cover letter based on the instruction below.

Candidate: {candidate_name}
Job Title: {job_title}
Company: {company_name}

Instruction: {instruction}

Current cover letter:
{cover_letter}

Return ONLY the updated cover letter text. Keep the same general structure unless the instruction says otherwise. Do not add placeholders.""")
        ])

        response = invoke_with_fallback(prompt, {
            "candidate_name": request.context.get("candidate_name", ""),
            "job_title": request.context.get("job_title", ""),
            "company_name": request.context.get("company_name", ""),
            "instruction": request.instruction,
            "cover_letter": request.cover_letter
        })

        return {"cover_letter": response.content.strip()}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Refinement failed: {str(e)}"
        )

    