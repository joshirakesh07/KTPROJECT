import os
import io
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


# ==================================================
# GOOGLE API
# ==================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing")


# ==================================================
# GEMINI
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ==================================================
# KT KNOWLEDGE
# ==================================================

KT_KNOWLEDGE = """
You are KT, an AI Career Guidance Agent.

Analyze the student's resume information.

Identify:
- Education
- Skills
- Programming languages
- Technologies
- Projects
- Internships
- Certifications
- GitHub

Determine career readiness.

Identify important skill gaps.

Recommend suitable fresher jobs and internships.

Recommend one practical project.

Suggest GitHub improvements.

Create a practical 30-day career plan.

Give one clear final recommendation.

Use only information provided in the resume.
Do not invent qualifications or experience.
"""


# ==================================================
# INPUT MODEL
# ==================================================

class AgentInput(BaseModel):
    input: str


# ==================================================
# KT FUNCTION
# ==================================================

def run_kt(resume_text: str):

    prompt = f"""
{KT_KNOWLEDGE}

========================
RESUME INFORMATION
========================

{resume_text}

========================
OUTPUT
========================

Give the result in this format:

CAREER READINESS:
...

STRONG SKILLS:
...

SKILL GAPS:
...

SUITABLE JOB ROLES:
...

RECOMMENDED PROJECT:
...

GITHUB IMPROVEMENTS:
...

30-DAY PLAN:
...

FINAL RECOMMENDATION:
...
"""

    print("Calling Gemini...")

    response = llm.invoke(prompt)

    print("Gemini completed.")

    return response.content


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="KT Career Agent"
)


# ==================================================
# SIMPLE JSON ENDPOINT
# ==================================================

@app.post("/agent")
def agent(data: AgentInput):

    try:

        result = run_kt(data.input)

        return {
            "output": result
        }

    except Exception as e:

        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==================================================
# PDF ENDPOINT
# ==================================================

@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:

        pdf_bytes = await file.read()

        pdf_file = io.BytesIO(pdf_bytes)

        from pypdf import PdfReader

        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():

            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF"
            )

        result = run_kt(text)

        return {
            "status": "success",
            "filename": file.filename,
            "output": result
        }

    except HTTPException:

        raise

    except Exception as e:

        print("PDF ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "agent": "KT Career Agent",
        "agent_endpoint": "/agent",
        "pdf_endpoint": "/analyze-pdf"
    }


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==================================================
# MAIN
# ==================================================

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
