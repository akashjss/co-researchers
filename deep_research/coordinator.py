from typing import Dict, Any
from .agents import *

class DeepResearcher:
    def __init__(self):
        self.initial_researcher = InitialResearchAgent()
        self.deep_diver = DeepDiveAgent()
        self.analyzer = AnalysisAgent()
        self.fact_checker = FactCheckAgent()
        self.critic = CriticalReviewAgent()
        self.synthesizer = SynthesisAgent()
        self.recommender = RecommendationAgent()

    async def research(self, topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform deep research on a topic
        
        Args:
            topic: The research topic or question
            depth: Research depth ("brief", "comprehensive", or "exhaustive")
        """
        # Initial research framework
        framework = await self.initial_researcher.agent.arun(
            f"Create a research framework for {depth} investigation of: {topic}"
        )

        # Deep dive research
        deep_dive = await self.deep_diver.agent.arun(
            f"Conduct detailed research based on this framework:\n{framework.content}"
        )

        # Analysis of findings
        analysis = await self.analyzer.agent.arun(
            f"Analyze these research findings:\n{deep_dive.content}"
        )

        # Fact checking
        fact_check = await self.fact_checker.agent.arun(
            f"Verify the key claims and findings:\n{deep_dive.content}\n\nAnalysis:\n{analysis.content}"
        )

        # Critical review
        critique = await self.critic.agent.arun(
            f"Critically review the research and analysis:\n{deep_dive.content}\n\nAnalysis:\n{analysis.content}"
        )

        # Synthesis of all findings
        synthesis = await self.synthesizer.agent.arun(f"""
        Synthesize all research components:
        
        Framework: {framework.content}
        Deep Dive: {deep_dive.content}
        Analysis: {analysis.content}
        Fact Check: {fact_check.content}
        Critique: {critique.content}
        """)

        # Recommendations
        recommendations = await self.recommender.agent.arun(
            f"Provide recommendations based on the synthesis:\n{synthesis.content}"
        )

        return {
            "framework": framework.content,
            "deep_dive": deep_dive.content,
            "analysis": analysis.content,
            "fact_check": fact_check.content,
            "critique": critique.content,
            "synthesis": synthesis.content,
            "recommendations": recommendations.content
        } 