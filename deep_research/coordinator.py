from typing import Dict, Any
import asyncio
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
        # Add delays between API calls to respect rate limits
        async def rate_limited_call(agent, prompt):
            await asyncio.sleep(0.25)  # Wait 250ms between calls to stay under 5 req/sec
            return await agent.arun(prompt)

        # Initial research framework
        framework = await rate_limited_call(
            self.initial_researcher.agent,
            f"Create a research framework for {depth} investigation of: {topic}"
        )

        # Deep dive research
        deep_dive = await rate_limited_call(
            self.deep_diver.agent,
            f"Conduct detailed research based on this framework:\n{framework.content}"
        )

        # Analysis of findings
        analysis = await rate_limited_call(
            self.analyzer.agent,
            f"Analyze these research findings:\n{deep_dive.content}"
        )

        # Fact checking
        fact_check = await rate_limited_call(
            self.fact_checker.agent,
            f"Verify the key claims and findings:\n{deep_dive.content}\n\nAnalysis:\n{analysis.content}"
        )

        # Critical review
        critique = await rate_limited_call(
            self.critic.agent,
            f"Critically review the research and analysis:\n{deep_dive.content}\n\nAnalysis:\n{analysis.content}"
        )

        # Synthesis of all findings
        synthesis = await rate_limited_call(
            self.synthesizer.agent,
            f"""
            Synthesize all research components:
            
            Framework: {framework.content}
            Deep Dive: {deep_dive.content}
            Analysis: {analysis.content}
            Fact Check: {fact_check.content}
            Critique: {critique.content}
            """
        )

        # Recommendations
        recommendations = await rate_limited_call(
            self.recommender.agent,
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