

import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field


# --------------------------------------------------
# GOOGLE API KEY
# --------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# --------------------------------------------------
# GEMINI
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# --------------------------------------------------
# INPUT MODEL
# --------------------------------------------------

class AgentInput(BaseModel):
    input: str = Field(
        description="Resume text to analyze"
    )


# --------------------------------------------------
# PDF / RESUME PARSING TOOL
# --------------------------------------------------

@tool
def parse_resume(resume_text: str) -> str:
    """
    Parse resume information and extract important
    student career information.
    """

    print("=" * 60)
    print("PDF PARSING TOOL CALLED")
    print("=" * 60)

    prompt = f"""
You are an AI Resume PDF Parsing Agent.

Analyze the following resume:

{resume_text}

Extract the following information:

1. Student Name
2. Email
3. Phone Number
4. Education
5. Skills
6. Programming Languages
7. Technologies
8. Frameworks
9. Databases
10. Tools
11. Projects
12. Internships
13. Work Experience
14. Certifications
15. Achievements
16. GitHub URL
17. LinkedIn URL

Rules:

- Do not invent information.
- If information is missing, write "Not mentioned".
- Keep technical terms accurate.
- Extract all important projects.
- Extract all important skills.
- Give a short resume summary.
- Suggest suitable career areas based only on the resume.

Return the answer in this format:

STUDENT INFORMATION
-------------------
Name:
Email:
Phone:

EDUCATION
---------
Education:

SKILLS
------
Skills:

PROGRAMMING LANGUAGES
---------------------
Languages:

TECHNOLOGIES
------------
Technologies:

FRAMEWORKS
----------
Frameworks:

DATABASES
---------
Databases:

TOOLS
-----
Tools:

PROJECTS
--------
Projects:

INTERNSHIPS
-----------
Internships:

WORK EXPERIENCE
---------------
Experience:

CERTIFICATIONS
--------------
Certifications:

ACHIEVEMENTS
------------
Achievements:

GITHUB
------
GitHub URL:

LINKEDIN
--------
LinkedIn URL:

KEYWORDS
--------
Important keywords:

RESUME SUMMARY
--------------
Resume summary:

CAREER PROFILE
--------------
Suitable career roles:
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# AGENT FUNCTION
# --------------------------------------------------

def resume_agent(data: AgentInput):

    print("=" * 60)
    print("RESUME AGENT")
    print("=" * 60)

    resume_text = data.input

    result = parse_resume.invoke(
        resume_text
    )

    return result


# --------------------------------------------------
# LANGCHAIN CHAIN
# --------------------------------------------------

chain = RunnableLambda(
    resume_agent
)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="AI Resume PDF Parsing Agent",
    description="Resume Parsing Agent using Google Gemini",
    version="1.0"
)


# --------------------------------------------------
# LANGSERVE
# --------------------------------------------------
# IMPORTANT:
# This creates:
# /agent/invoke
# /agent/batch
# /agent/stream
# /agent/playground/
# --------------------------------------------------

add_routes(
    app,
    chain.with_types(
        input_type=AgentInput
    ),
    path="/agent",
    playground_type="default"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "AI Resume PDF Parsing Agent Running",
        "agent": "PDF Parsing Agent",
        "model": "gemini-2.5-flash",
        "langserve_endpoint": "/agent",
        "playground": "/agent/playground/"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "PDF Parsing Agent",
        "gemini": "connected"
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

@app.get("/test")
def test():

    return {
        "message": "PDF Parsing Agent is working",
        "endpoint": "/agent",
        "playground": "/agent/playground/"
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
