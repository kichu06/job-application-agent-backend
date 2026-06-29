from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    extract_jd_skills,
    parse_resume,
    analyze_gaps,
    generate_cover_letter,
    generate_interview_questions
)

def create_agent():
    # 1. Create the graph with our state
    graph = StateGraph(AgentState)

    # 2. Add all nodes
    graph.add_node("extract_jd_skills", extract_jd_skills)
    graph.add_node("parse_resume", parse_resume)
    graph.add_node("analyze_gaps", analyze_gaps)
    graph.add_node("generate_cover_letter", generate_cover_letter)
    graph.add_node("generate_interview_questions", generate_interview_questions)

    # 3. Define the flow — edges between nodes
    graph.set_entry_point("extract_jd_skills")
    graph.add_edge("extract_jd_skills", "parse_resume")
    graph.add_edge("parse_resume", "analyze_gaps")
    graph.add_edge("analyze_gaps", "generate_cover_letter")
    graph.add_edge("generate_cover_letter", "generate_interview_questions")
    graph.add_edge("generate_interview_questions", END)

    # 4. Compile and return
    return graph.compile()


# The agent instance — imported by main.py
agent = create_agent()