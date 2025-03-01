from typing import List, Dict, Any
from .agents import *

class AICoScientist:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.generator = GenerationAgent()
        self.reflector = ReflectionAgent()
        self.ranker = RankingAgent()
        self.evolver = EvolutionAgent()
        self.proximity = ProximityAgent()
        self.meta_reviewer = MetaReviewAgent()
        
    async def research(self, goal: str) -> Dict[str, Any]:
        """
        Execute the research process for a given goal
        """
        # 1. Supervisor creates research plan
        plan = await self.supervisor.agent.arun(
            f"Create a structured research plan for the following goal: {goal}"
        )
        
        # 2. Generator explores and creates hypotheses
        hypotheses = await self.generator.agent.arun(
            f"Generate initial hypotheses for: {goal}\n\nResearch Plan:\n{plan.content}"
        )
        
        # 3. Reflector reviews hypotheses
        reviews = await self.reflector.agent.arun(
            f"Review these hypotheses:\n{hypotheses.content}"
        )
        
        # 4. Ranker creates tournament rankings
        rankings = await self.ranker.agent.arun(
            f"Create pairwise rankings for these reviewed hypotheses:\n{reviews.content}"
        )
        
        # 5. Evolver refines top hypotheses
        evolved = await self.evolver.agent.arun(
            f"Refine the top ranked hypotheses:\n{rankings.content}"
        )
        
        # 6. Proximity agent groups similar hypotheses
        grouped = await self.proximity.agent.arun(
            f"Group similar hypotheses:\n{evolved.content}"
        )
        
        # 7. Meta-reviewer generates final report
        report = await self.meta_reviewer.agent.arun(
            f"Generate a comprehensive report synthesizing all findings:\n{grouped.content}"
        )
        
        return {
            "plan": plan.content,
            "hypotheses": hypotheses.content, 
            "reviews": reviews.content,
            "rankings": rankings.content,
            "evolved": evolved.content,
            "grouped": grouped.content,
            "report": report.content
        } 