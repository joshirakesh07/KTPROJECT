

import os
import uvicorn
import requests
import json

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field


# ==================================================
# GOOGLE API KEY
# ==================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# ==================================================
# GEMINI
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ==================================================
# INPUT MODEL
# ==================================================

class AgentInput(BaseModel):

    student_name: str = Field(
        description="Student name"
    )

    email: str = Field(
        default="Not mentioned"
    )

    phone: str = Field(
        default="Not mentioned"
    )

    education: str = Field(
        default="Not mentioned"
    )

    skills: list[str] = Field(
        default=[]
    )

    programming_languages: list[str] = Field(
        default=[]
    )

    technologies: list[str] = Field(
        default=[]
    )

    frameworks: list[str] = Field(
        default=[]
    )

    databases: list[str] = Field(
        default=[]
    )

    projects: list[str] = Field(
        default=[]
    )

    internships: list[str] = Field(
        default=[]
    )

    certifications: list[str] = Field(
        default=[]
    )

    achievements: list[str] = Field(
        default=[]
    )

    github: str = Field(
        default=""
    )

    linkedin: str = Field(
        default=""
    )

    target_role: str = Field(
        default="AI/ML Engineer"
    )


# ==================================================
# TOOL 1 - JOB SEARCH
# ==================================================

@tool
def job_search(role: str) -> str:
    """
    Find suitable job and internship opportunities
    for the student's target role.
    """

    print("=" * 60)
    print("JOB SEARCH TOOL")
    print("ROLE:", role)
    print("=" * 60)

    prompt = f"""
You are a job search assistant.

Target role:
{role}

Find suitable opportunities for:

- Freshers
- Entry-level candidates
- Internships
- Graduate roles

Focus on India.

Provide useful job-search recommendations.

Include:
1. Job title
2. Company if known
3. Required skills
4. Experience
5. Location
6. Application/source link if available

IMPORTANT:
Do not invent current job openings.
If live job information is unavailable, provide
search directions instead.
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# TOOL 2 - SKILL GAP
# ==================================================

@tool
def analyze_skill_gap(
    resume_information: str,
    target_role: str
) -> str:
    """
    Compare the student's skills with the target role.
    """

    print("=" * 60)
    print("SKILL GAP TOOL")
    print("TARGET ROLE:", target_role)
    print("=" * 60)

    prompt = f"""
You are an AI career skill-gap analyzer.

Target role:
{target_role}

Student resume information:
{resume_information}

Analyze the student.

Identify:

1. Current skills
2. Matching skills
3. Missing skills
4. Skills that need improvement
5. High-priority skills
6. Recommended learning order

Do not claim that the student has skills
that are not present in the provided information.

Give practical recommendations.
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# TOOL 3 - PROJECT RECOMMENDATION
# ==================================================

@tool
def project_recommendation(
    target_role: str,
    skill_gap: str
) -> str:
    """
    Recommend portfolio projects based on the
    student's target role and skill gaps.
    """

    print("=" * 60)
    print("PROJECT RECOMMENDATION TOOL")
    print("=" * 60)

    prompt = f"""
You are a technical project mentor.

Target role:
{target_role}

Skill gap:
{skill_gap}

Recommend 5 practical projects.

For each project provide:

1. Project title
2. Problem statement
3. Technologies
4. Main features
5. Skills demonstrated
6. Difficulty
7. Why it helps for the target role

Projects should be realistic for a B.Tech student.
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# TOOL 4 - GITHUB CHECK
# ==================================================

@tool
def github_check(username: str) -> str:
    """
    Analyze a student's public GitHub repositories.
    """

    print("=" * 60)
    print("GITHUB TOOL")
    print("USERNAME:", username)
    print("=" * 60)

    if not username:

        return "GitHub username was not provided."

    username = username.strip()

    if "github.com/" in username:

        username = username.rstrip(
            "/"
        ).split(
            "github.com/"
        )[-1]

    url = f"https://api.github.com/users/{username}/repos"

    try:

        response = requests.get(
            url,
            params={
                "sort": "updated",
                "per_page": 10
            },
            timeout=15
        )

        if response.status_code == 404:

            return "GitHub user not found."

        if response.status_code != 200:

            return (
                "GitHub API error: "
                + str(response.status_code)
            )

        repositories = response.json()

        if not repositories:

            return "No public repositories found."

        repo_information = []

        for repo in repositories:

            repo_information.append({
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get(
                    "stargazers_count",
                    0
                ),
                "forks": repo.get(
                    "forks_count",
                    0
                ),
                "url": repo.get("html_url"),
                "updated": repo.get("updated_at")
            })

        return json.dumps(
            repo_information,
            indent=2
        )

    except Exception as e:

        return f"GitHub check failed: {str(e)}"


# ==================================================
# KT AGENT CORE
# ==================================================

def career_agent(data: AgentInput):

    print("=" * 60)
    print("KT CAREER AGENT")
    print("=" * 60)

    # ------------------------------------------------
    # Convert PDF Agent information to text
    # ------------------------------------------------

    resume_information = json.dumps(
        {
            "student_name": data.student_name,
            "email": data.email,
            "phone": data.phone,
            "education": data.education,
            "skills": data.skills,
            "programming_languages":
                data.programming_languages,
            "technologies": data.technologies,
            "frameworks": data.frameworks,
            "databases": data.databases,
            "projects": data.projects,
            "internships": data.internships,
            "certifications": data.certifications,
            "achievements": data.achievements,
            "github": data.github,
            "linkedin": data.linkedin
        },
        indent=2
    )

    target_role = data.target_role


    # ------------------------------------------------
    # INITIAL PROFILE ANALYSIS
    # ------------------------------------------------

    profile_prompt = f"""
You are the main KT Career Agent.

Analyze this structured resume information.

Resume:
{resume_information}

Target Role:
{target_role}

Give a short profile analysis containing:

1. Student profile
2. Strongest skills
3. Relevant projects
4. Current career readiness
"""

    profile = llm.invoke(
        profile_prompt
    ).content


    # ------------------------------------------------
    # JOB SEARCH TOOL
    # ------------------------------------------------

    jobs = job_search.invoke(
        target_role
    )


    # ------------------------------------------------
    # SKILL GAP TOOL
    # ------------------------------------------------

    skill_gap = analyze_skill_gap.invoke(
        {
            "resume_information":
                resume_information,

            "target_role":
                target_role
        }
    )


    # ------------------------------------------------
    # PROJECT TOOL
    # ------------------------------------------------

    projects = project_recommendation.invoke(
        {
            "target_role":
                target_role,

            "skill_gap":
                skill_gap
        }
    )


    # ------------------------------------------------
    # GITHUB TOOL
    # ------------------------------------------------

    github = github_check.invoke(
        data.github
    )


    # ------------------------------------------------
    # FINAL SYNTHESIS
    # ------------------------------------------------

    final_prompt = f"""
You are the Final KT Career Advisor.

Use the following information.

================================
STUDENT
================================

{resume_information}

Target Role:
{target_role}


================================
PROFILE ANALYSIS
================================

{profile}


================================
JOB SEARCH
================================

{jobs}


================================
SKILL GAP
================================

{skill_gap}


================================
PROJECT RECOMMENDATIONS
================================

{projects}


================================
GITHUB
================================

{github}


================================
FINAL TASK
================================

Create a practical career recommendation.

Include:

1. Career Readiness
2. Strong Skills
3. Skill Gaps
4. Best Project to Build
5. Suitable Job Roles
6. Job Search Direction
7. GitHub Improvements
8. 30-Day Action Plan
9. Final Recommendation

Do not invent information.

Clearly separate:
- What the student already has
- What the student needs to learn
- What the student should do next

Keep the answer practical and suitable
for a college student/fresher.
"""

    final_result = llm.invoke(
        final_prompt
    ).content


    # ------------------------------------------------
    # RETURN
    # ------------------------------------------------

    return {

        "student_name":
            data.student_name,

        "target_role":
            target_role,

        "profile_analysis":
            profile,

        "job_search":
            jobs,

        "skill_gap":
            skill_gap,

        "project_recommendations":
            projects,

        "github_analysis":
            github,

        "final_synthesis":
            final_result
    }


# ==================================================
# LANGCHAIN CHAIN
# ==================================================

chain = RunnableLambda(
    career_agent
)


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(

    title="KT AI Career Agent",

    description=(
        "Career Agent that receives structured "
        "PDF Agent information"
    ),

    version="2.0"
)


# ==================================================
# LANGSERVE
# ==================================================

add_routes(

    app,

    chain.with_types(
        input_type=AgentInput
    ),

    path="/agent",

    playground_type="default"
)


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return {

        "status": "running",

        "message":
            "KT AI Career Agent Running",

        "input":
            "Structured PDF Agent Information",

        "model":
            "gemini-2.5-flash",

        "endpoint":
            "/agent",

        "playground":
            "/agent/playground/"
    }


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "agent":
            "KT Career Agent",

        "gemini":
            "connected"
    }


# ==================================================
# TEST
# ==================================================

@app.get("/test")
def test():

    return {

        "message":
            "KT Career Agent is working",

        "input":
            "PDF Agent structured information",

        "endpoint":
            "/agent",

        "playground":
            "/agent/playground/"
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
