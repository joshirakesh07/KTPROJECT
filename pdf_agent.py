

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
        description="Resume text to analyze"
    )


# --------------------------------------------------
# TOOL
# --------------------------------------------------

@tool
def parse_resume(resume_text: str) -> str:
    """
    Parse resume information.
    """

    prompt = f"""
You are an AI Resume PDF Parsing Agent.

Analyze the following resume:

{resume_text}

Extract:

1. Student Name
2. Email
3. Phone
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

Do not invent information.

If information is missing, say:
Not mentioned.

Return a clear structured answer.
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# AGENT
# --------------------------------------------------

def pdf_agent(data: PDFInput):

    print("PDF AGENT CALLED")

    result = parse_resume.invoke(
        data.input
    )

    return result


# --------------------------------------------------
# CHAIN
# --------------------------------------------------

chain = RunnableLambda(pdf_agent)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="PDF Resume Parsing Agent"
)


# --------------------------------------------------
# LANGSERVE ROUTE
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
        "message": "PDF Resume Parsing Agent Running"
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
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
