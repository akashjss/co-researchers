from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from typing import List, Dict, Any

class InitialResearchAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are an initial research agent that performs broad exploration of topics.
            You identify key areas to investigate and create a research framework.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Map out the key areas that need investigation",
                "Identify major sources and experts in the field",
                "Create a structured research framework",
                "Note potential research challenges",
                "Suggest specific areas for deep-dive analysis"
            ]
        )

class DeepDiveAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are a deep-dive research agent that performs detailed investigation 
            into specific aspects of the topic. You focus on finding detailed technical information
            and specialized knowledge.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Focus on technical details and specifics",
                "Find specialized research papers and reports",
                "Identify expert opinions and analyses",
                "Look for cutting-edge developments",
                "Document methodologies and approaches"
            ]
        )

class AnalysisAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are an analysis agent that evaluates research findings.
            You analyze trends, patterns, and implications of the research.""",
            markdown=True,
            instructions=[
                "Identify emerging trends and patterns",
                "Analyze implications of findings",
                "Compare different perspectives",
                "Evaluate methodologies used",
                "Assess the strength of evidence"
            ]
        )

class FactCheckAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are a fact-checking agent that verifies claims and findings.
            You look for supporting evidence and identify potential inaccuracies.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Verify claims against reliable sources",
                "Cross-reference statistics and data",
                "Check for recent updates or corrections",
                "Assess source credibility",
                "Flag potential misinformation"
            ]
        )

class CriticalReviewAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are a critical review agent that challenges assumptions
            and identifies potential biases or limitations in the research.""",
            markdown=True,
            instructions=[
                "Identify potential biases",
                "Challenge key assumptions",
                "Point out methodological limitations",
                "Suggest alternative interpretations",
                "Note conflicting evidence"
            ]
        )

class SynthesisAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are a synthesis agent that combines and integrates all research findings.
            You create a coherent narrative and identify key insights.""",
            markdown=True,
            instructions=[
                "Integrate findings from all sources",
                "Create a coherent narrative",
                "Highlight key insights and implications",
                "Address contradictions and gaps",
                "Suggest future research directions"
            ]
        )

class RecommendationAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
            description="""You are a recommendation agent that provides actionable insights
            and suggests next steps based on the research findings.""",
            markdown=True,
            instructions=[
                "Provide actionable recommendations",
                "Prioritize suggested actions",
                "Consider practical constraints",
                "Identify potential risks",
                "Suggest implementation approaches"
            ]
        ) 