from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from typing import List, Dict, Any

class SupervisorAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are a supervisor agent that orchestrates the research process, assigns tasks to other agents, and allocates resources based on the research plan.",
            tools=[ExaTools()],
            markdown=True
        )

class GenerationAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are a generation agent that creates initial research hypotheses by exploring literature, simulating debates, and identifying testable assumptions.",
            tools=[ExaTools()],
            markdown=True
        )

class ReflectionAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are a reflection agent that reviews hypotheses, assesses correctness, quality, novelty, and potential to explain existing observations.",
            tools=[ExaTools()],
            markdown=True
        )

class RankingAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are a ranking agent that creates pairwise comparisons of hypotheses using simulated debates to create an Elo rating.",
            markdown=True
        )

class EvolutionAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are an evolution agent that refines best hypotheses by grounding them in literature, improving coherence/feasibility, combining ideas and exploring out-of-the-box thinking.",
            tools=[ExaTools()],
            markdown=True
        )

class ProximityAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are a proximity agent that groups similar hypotheses to optimize exploration diversity.",
            markdown=True
        )

class MetaReviewAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="You are a meta-review agent that synthesizes insights from all reviews, identifies patterns, optimizes other agents' performance, and creates reports.",
            markdown=True
        ) 