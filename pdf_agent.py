

import os
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from langserve import add_routes

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field

from pypdf import PdfReader


# ============================================================
# GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing")


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Resume information extracted from PDF"
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file):

    try:

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():

            raise ValueError(
                "No readable text found in PDF"
            )

        return text

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"PDF reading error: {str(e)}"
        )


# ============================================================
# KT CAREER AGENT
# ============================================================

def kt_agent(data: AgentInput):

    print("=" * 60)
    print("KT AGENT STARTED")
    print("=" * 60)

    resume_text = data.input

    prompt = f"""

You are KT, an AI Career Agent.

The following information was extracted
from a student's PDF resume.

================ RESUME =================

{resume_text}

===========================================

Analyze the student's resume.

Provide the following:

1. Career Readiness
   - Give a short assessment of the student's
     current career readiness.

2. Education
   - Identify the education information.

3. Strong Skills
   - List the strongest technical skills.

4. Skill Gaps
   - Identify important missing skills
     based on the student's current profile.

5. Suitable Job Roles
   - Recommend suitable fresher and
     entry-level job roles.

6. Project Recommendation
   - Recommend ONE practical project
     that will improve the student's profile.

7. GitHub Improvement
   - Suggest improvements to the GitHub
     profile and repositories if GitHub
     information is available.

8. Internship / Job Recommendation
   - Suggest what type of internships
     or jobs the student should target.

9. 30-Day Career Plan
   - Give a simple 30-day action plan.

10. Final Recommendation
   - Give the single most valuable next
     action for the student.

IMPORTANT RULES:

- Use only information available in the resume.
- Do not invent qualifications.
- Clearly say "Not mentioned" when information
  is unavailable.
- Keep the recommendations practical.
- Prioritize the most valuable next action.
- Do not give a long unrelated list.

"""

    print("SENDING REQUEST TO GEMINI...")

    try:

        response = llm.invoke(prompt)

        print("GEMINI RESPONSE RECEIVED")

        return response.content

    except Exception as e:

        print("GEMINI ERROR:", str(e))

        return {
            "error": str(e)
        }


# ============================================================
# LANGCHAIN CHAIN
# ============================================================

chain = RunnableLambda(kt_agent)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="KT PDF Career Agent",
    description="PDF Resume Parser and KT Career Analysis Agent"
)


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    chain.with_types(
        input_type=AgentInput
    ),
    path="/agent",
    playground_type="default"
)


# ============================================================
# PDF UPLOAD ENDPOINT
# ============================================================

@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...)
):

    print("=" * 60)
    print("PDF RECEIVED")
    print("FILE:", file.filename)
    print("=" * 60)

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:

        pdf_bytes = await file.read()

        import io

        pdf_file = io.BytesIO(pdf_bytes)

        # --------------------------------------------
        # Extract PDF text
        # --------------------------------------------

        resume_text = extract_pdf_text(
            pdf_file
        )

        print("PDF TEXT EXTRACTED")
        print("TEXT LENGTH:", len(resume_text))

        # --------------------------------------------
        # Send to KT Agent
        # --------------------------------------------

        result = kt_agent(
            AgentInput(
                input=resume_text
            )
        )

        return {
            "filename": file.filename,
            "status": "success",
            "career_analysis": result
        }

    except Exception as e:

        print("PDF AGENT ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "agent": "KT PDF Career Agent",
        "pdf_endpoint": "/analyze-pdf",
        "text_endpoint": "/agent",
        "playground": "/agent/playground/"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "KT PDF Career Agent"
    }


# ============================================================
# MAIN
# ============================================================

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
