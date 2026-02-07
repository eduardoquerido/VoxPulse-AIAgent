import os
from dotenv import load_dotenv

load_dotenv()

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import TavilySearchTool
from models.models import ComparisonOutput

# from langchain_google_genai import ChatGoogleGenerativeAI

@CrewBase
class VoxPulseCrew:
    """VoxPulse-AI Multi-Agent System for Political Analysis"""

    def __init__(self) -> None:
        # Configuration for the LLM
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash",
        #     verbose=True,
        #     temperature=0.3,  # low value to avoid hallucination
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )
        self.llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
        self.search_tool = TavilySearchTool()

    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Political Data Researcher",
            goal="Find the latest and most relevant news about {politician}",
            backstory="Expert in digital monitoring for the 2026 election cycle.",
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            role="Political Sentiment Analyst",
            goal=(
                    "Analyze the sentiment of the data and categorize by theme, "
                    "including Public Security, Economy Trust, and Social Approval."
                ),
            backstory="Specialist in linguistics and Brazilian political science.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            description="Search for the top 100 recent news regarding {politician} in Brazil.",
            expected_output="A bullet-point list of the most relevant events.",
            agent=self.researcher(),
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            description="Analyze the sentiment (0-100) and identify themes (Economy, Health, Public Security, Social Approval).",
            expected_output="A structured report with Sentiment Score, Topics, and Crisis Alerts written entirely in {language}.",
            agent=self.analyst(),
        )

    @task
    def comparison_task(self) -> Task:
        return Task(
            description="Analyze and compare the following politicians: {politicians_list}.",
            expected_output="A structured JSON object with sentiment and trust metrics for each name.",
            output_json=ComparisonOutput,
            agent=self.analyst(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the VoxPulse-AI crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=3,
            allow_delegation=False,
        )

def run_analysis(main_politician: str, all_politicians: str, target_lang: str):
    crew_output = (
        VoxPulseCrew()
        .crew()
        .kickoff(
            inputs={
                "politician": main_politician,
                "politicians_list": all_politicians,
                "language": target_lang,
            }
        )
    )

    return crew_output
