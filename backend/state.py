from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Inputs
    jd_text: str
    resume_text: str  
    language: str 

    # Node 1 output
    extracted_skills: Optional[dict]

    # Node 2 output
    parsed_resume: Optional[dict]

    # Node 3 output
    skill_gaps: Optional[dict]

    # Node 4 output
    cover_letter: Optional[str]

    # Node 5 output
    interview_questions: Optional[list]