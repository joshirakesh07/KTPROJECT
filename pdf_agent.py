

import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field


# ================================================
# GOOGLE API KEY
# ================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing")


# ================================================
# GEMINI
# ================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ================================================
# INPUT
# ================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Information received from PDF Agent"
    )


# ================================================
# KT AGENT
# ================================================

def kt_agent(data: AgentInput):

    print("KT AGENT STARTED")

    prompt = f"""
You are KT, an AI Career Agent.

The following information was extracted
from a student's PDF resume:

{data.input}

Analyze the information and provide:

1. Career readiness
2. Strong skills
3. Skill gaps
4. Suitable job roles
5. One recommended project
6. GitHub improvements
7. A short 30-day action plan

Do not invent information.

Keep the response practical and concise.
"""

    print("SENDING REQUEST TO GEMINI")

    response = llm.invoke(prompt)

    print("GEMINI RESPONSE RECEIVED")

    return response.content


# ================================================
# CHAIN
# ================================================

chain = RunnableLambda(kt_agent)


# ================================================
# FASTAPI
# ================================================

app = FastAPI(
    title="KT Career Agent"
)


# ================================================
# LANGSERVE
# ================================================

add_routes(
    app,
    chain.with_types(
        input_type=AgentInput
    ),
    path="/agent",
    playground_type="default"
)


# ================================================
# HOME
# ================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "agent": "KT Career Agent",
        "endpoint": "/agent",
        "playground": "/agent/playground/"
    }


# ================================================
# HEALTH
# ================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ================================================
# MAIN
# ================================================

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
