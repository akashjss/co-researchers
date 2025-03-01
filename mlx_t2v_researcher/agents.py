# Placeholder for future agent implementations
from datetime import datetime
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools

def create_base_agent(name: str, system_prompt: str) -> Agent:
    """Helper function to create agents with consistent configuration"""
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),  # Updated model ID
        description=dedent(f"""
        You are {name}, an expert AI agent specializing in MLX model conversion.
        {system_prompt}
        Your responses should be:
        - Clear and technical
        - Implementation-focused
        - Well-structured
        - Backed by documentation
        """),
        instructions=dedent("""
        - Analyze the provided context thoroughly
        - Reference relevant documentation when available
        - Provide practical, implementable solutions
        - Consider Apple Silicon optimizations
        - Include code examples where appropriate
        """),
        tools=[ExaTools()],  # Adding Exa search capabilities
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True
    ) 