from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    jd_text: str = Form(...)
):
    # Validate file is a PDF
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Validate JD text is not empty
    if not jd_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty"
        )

    # Step 1 — Extract text from PDF
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

    # Step 2 — Run LangGraph agent
    try:
        result = agent.invoke({
            "jd_text": jd_text,
            "resume_text": resume_text,
            "extracted_skills": None,
            "parsed_resume": None,
            "skill_gaps": None,
            "cover_letter": None,
            "interview_questions": None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in terminal
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}"
        )

    # Step 3 — Return results
    return {
        "extracted_skills": result["extracted_skills"],
        "parsed_resume": result["parsed_resume"],
        "skill_gaps": result["skill_gaps"],
        "cover_letter": result["cover_letter"],
        "interview_questions": result["interview_questions"]
    }