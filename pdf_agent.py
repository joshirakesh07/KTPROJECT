
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

class PDFInput(BaseModel):

    input: str = Field(
        description="Resume text or PDF extracted text"
    )


# --------------------------------------------------
# TOOL
# --------------------------------------------------

@tool
def parse_resume(resume_text: str) -> str:
    """
    Parse resume information from PDF text.
    """

    print("=" * 50)
    print("PDF PARSING TOOL CALLED")
    print("RESUME RECEIVED")
    print("=" * 50)

    prompt = f"""
You are an AI Resume PDF Parsing Agent.

Analyze the resume information below.

Resume:
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
10. Projects
11. Internships
12. Work Experience
13. Certifications
14. Achievements
15. GitHub
16. LinkedIn

Rules:

- Do not invent information.
- If information is missing, say "Not mentioned".
- Keep technical names accurate.
- Extract all important projects.
- Extract all important skills.

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

def pdf_agent(data: PDFInput):

    resume_text = data.input

    result = parse_resume.invoke(
        resume_text
    )

    return result


# --------------------------------------------------
# CHAIN
# --------------------------------------------------

chain = RunnableLambda(
    pdf_agent
)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="PDF Resume Parsing Agent",
    description="AI PDF Resume Parsing Agent using Google Gemini",
    version="1.0"
)


# --------------------------------------------------
# LANGSERVE
# --------------------------------------------------

add_routes(
    app,
    chain.with_types(
        input_type=PDFInput
    ),
    path="/pdf-agent",
    playground_type="default"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "PDF Resume Parsing Agent Running",
        "agent": "PDF Parsing Agent",
        "model": "Gemini 2.5 Flash",
        "endpoint": "/pdf-agent"
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "PDF Parsing Agent",
        "gemini": "connected"
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8001
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
