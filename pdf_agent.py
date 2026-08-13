

import os
import uvicorn
import requests

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
# KT KNOWLEDGE
# ==================================================

KT_KNOWLEDGE = """

You are KT, an AI Career Guidance Agent.

Your purpose is to analyze information extracted from a student's
resume PDF and provide practical career recommendations.

You should analyze:

- Education
- Programming languages
- Technical skills
- Technologies
- Frameworks
- Databases
- Projects
- Internships
- Certifications
- Achievements
- GitHub
- LinkedIn
- Career objective or target role

CAREER READINESS:

Beginner:
Basic knowledge with little practical work.

Developing:
Relevant skills with some projects or coursework.

Job Ready:
Good technical foundation with relevant projects or experience.

Strong Candidate:
Strong skills, projects, experience and portfolio.

Do not treat this as an official hiring decision.

SKILL GAP:

Compare the student's existing skills with the requirements
of their target career.

Identify:
- Existing skills
- Strong skills
- Missing skills
- Skills that need improvement
- High-priority skills
- Recommended learning order

PROJECTS:

Recommend practical projects that match the student's target
role and skill gaps.

Projects should:
- Solve a real problem
- Use relevant technologies
- Demonstrate practical skills
- Be suitable for a college student
- Be useful for GitHub
- Be explainable in an interview

GITHUB:

Analyze the GitHub information if it is available.

Recommend:
- Better README files
- Screenshots
- Architecture diagrams
- Better repository names
- Project documentation
- Clean source code
- Project demonstrations
- Removal of unnecessary repositories

Never expose or request API keys, passwords or private credentials.

JOB SEARCH:

Recommend suitable:
- Fresher jobs
- Internships
- Graduate roles
- Entry-level positions
- Junior positions

Do not invent current job openings.

FINAL SYNTHESIS:

The final recommendation should contain:

1. Career Readiness
2. Strong Skills
3. Skill Gaps
4. Best Project Recommendation
5. Suitable Job Roles
6. Job Search Direction
7. GitHub Improvements
8. 30-Day Action Plan
9. Final Career Recommendation

Always distinguish between:
- Skills the student already has
- Skills the student needs to learn
- Actions the student should take next.

"""


# ==================================================
# INPUT MODEL
# ==================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Information extracted by the PDF Agent"
    )


# ==================================================
# TOOL 1 - JOB SEARCH
# ==================================================

@tool
def job_search(student_information: str) -> str:
    """
    Recommend suitable jobs and internships based on
    the student's resume information.
    """

    print("=" * 60)
    print("JOB SEARCH TOOL")
    print("=" * 60)

    prompt = f"""
You are a job search assistant.

Student information:

{student_information}

Based on the student's skills, education, projects
and career objective, identify suitable job categories.

Focus on:
- Fresher jobs
- Internships
- Entry-level jobs
- Graduate roles
- Junior positions

Provide:
1. Suitable job titles
2. Required skills
3. Why the student fits
4. Skills that need improvement

Do NOT invent specific current vacancies.
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# TOOL 2 - SKILL GAP
# ==================================================

@tool
def skill_gap(student_information: str) -> str:
    """
    Identify skill gaps from the student's resume.
    """

    print("=" * 60)
    print("SKILL GAP TOOL")
    print("=" * 60)

    prompt = f"""
You are a career skill-gap analyzer.

Student information:

{student_information}

Analyze the student's current skills.

Identify:

1. Existing skills
2. Strong skills
3. Missing skills
4. Skills that need improvement
5. High-priority skills
6. Recommended learning order

Do not invent skills.
Only use information available in the student data.

{KT_KNOWLEDGE}
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# TOOL 3 - PROJECT RECOMMENDATION
# ==================================================

@tool
def project_recommendation(
    student_information: str,
    skill_gap_result: str
) -> str:
    """
    Recommend projects based on the student's
    career goal and skill gaps.
    """

    print("=" * 60)
    print("PROJECT RECOMMENDATION TOOL")
    print("=" * 60)

    prompt = f"""
You are a technical project mentor.

Student information:

{student_information}

Skill gap:

{skill_gap_result}

Recommend 5 practical projects.

For each project provide:

1. Project title
2. Problem statement
3. Technologies
4. Main features
5. Skills demonstrated
6. Difficulty
7. Why it helps the student's career

Projects should be realistic for a B.Tech student.

{KT_KNOWLEDGE}
"""

    response = llm.invoke(prompt)

    return response.content


# ==================================================
# TOOL 4 - GITHUB CHECK
# ==================================================

@tool
def github_check(student_information: str) -> str:
    """
    Analyze the GitHub profile mentioned in the
    student's PDF information.
    """

    print("=" * 60)
    print("GITHUB CHECK TOOL")
    print("=" * 60)

    prompt = f"""
Find the GitHub username or GitHub URL from this
student information:

{student_information}

If no GitHub information is present, return:

GitHub information not available.

Otherwise return ONLY the GitHub username.
"""

    response = llm.invoke(prompt)

    username = response.content.strip()

    if (
        "not available" in username.lower()
        or "not mentioned" in username.lower()
    ):
        return "GitHub information not available."


    # ------------------------------------------------
    # Extract username
    # ------------------------------------------------

    if "github.com/" in username:

        username = username.split(
            "github.com/"
        )[-1]

    username = username.strip(
        "/ \n"
    )

    username = username.split()[0]


    # ------------------------------------------------
    # GitHub API
    # ------------------------------------------------

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

            return "GitHub profile found but no public repositories."

        result = []

        for repo in repositories:

            result.append({
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
                "url": repo.get("html_url")
            })

        return str(result)

    except Exception as e:

        return f"GitHub check failed: {str(e)}"


# ==================================================
# KT AGENT
# ==================================================

def kt_agent(data: AgentInput):

    print("=" * 60)
    print("KT CAREER AGENT")
    print("=" * 60)

    student_information = data.input


    # =================================================
    # PROFILE ANALYSIS
    # =================================================

    profile_prompt = f"""
You are KT, an AI Career Advisor.

Analyze this information extracted from a student's
resume PDF:

{student_information}

Identify:

1. Education
2. Current technical skills
3. Projects
4. Experience
5. Certifications
6. Career objective
7. GitHub information
8. Possible career direction

Do not invent information.

{KT_KNOWLEDGE}
"""

    profile = llm.invoke(
        profile_prompt
    ).content


    # =================================================
    # JOB SEARCH
    # =================================================

    jobs = job_search.invoke(
        student_information
    )


    # =================================================
    # SKILL GAP
    # =================================================

    gaps = skill_gap.invoke(
        student_information
    )


    # =================================================
    # PROJECTS
    # =================================================

    projects = project_recommendation.invoke(
        {
            "student_information":
                student_information,

            "skill_gap_result":
                gaps
        }
    )


    # =================================================
    # GITHUB
    # =================================================

    github = github_check.invoke(
        student_information
    )


    # =================================================
    # FINAL SYNTHESIS
    # =================================================

    final_prompt = f"""
You are KT, the final AI Career Advisor.

Use the following information.

========================================
STUDENT PDF INFORMATION
========================================

{student_information}


========================================
PROFILE ANALYSIS
========================================

{profile}


========================================
JOB SEARCH
========================================

{jobs}


========================================
SKILL GAP
========================================

{gaps}


========================================
PROJECT RECOMMENDATIONS
========================================

{projects}


========================================
GITHUB ANALYSIS
========================================

{github}


========================================
FINAL TASK
========================================

Create a final career recommendation.

Use this structure:

CAREER READINESS
----------------
Give Beginner / Developing / Job Ready /
Strong Candidate and explain why.

STRONG SKILLS
-------------
List the strongest existing skills.

SKILL GAPS
----------
List the most important missing skills.

BEST PROJECT TO BUILD
---------------------
Recommend the single best project.

SUITABLE JOB ROLES
------------------
List suitable roles.

JOB SEARCH DIRECTION
--------------------
Explain what types of jobs/internships to target.

GITHUB IMPROVEMENTS
-------------------
Give specific GitHub improvements.

30-DAY ACTION PLAN
------------------
Week 1:
Week 2:
Week 3:
Week 4:

FINAL RECOMMENDATION
--------------------
Give the most important next step.

IMPORTANT:
Do not invent information.
Clearly distinguish existing skills from recommended skills.
Keep the answer practical for a college student or fresher.
"""

    final_result = llm.invoke(
        final_prompt
    ).content


    # =================================================
    # RETURN
    # =================================================

    return final_result


# ==================================================
# CHAIN
# ==================================================

chain = RunnableLambda(
    kt_agent
)


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="KT AI Career Agent",
    description="KT Career Agent using PDF Agent information",
    version="1.0"
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
        "agent": "KT Career Agent",
        "input": "PDF Agent output",
        "model": "gemini-2.5-flash",
        "endpoint": "/agent",
        "playground": "/agent/playground/"
    }


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "KT Career Agent"
    }


# ==================================================
# TEST
# ==================================================

@app.get("/test")
def test():

    return {
        "message": "KT Agent is working",
        "input": "PDF Agent information"
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
