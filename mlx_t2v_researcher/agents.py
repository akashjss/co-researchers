from agno.agent import Agent, AgentMemory
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from agno.memory.db.postgres import PgMemoryDb
from typing import List, Dict, Any

class BaseMLXAgent:
    """Base class for MLX conversion agents with RAG capabilities"""
    def __init__(self, description: str, instructions: List[str], knowledge_base, memory_db_url: str):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description=description,
            tools=[ExaTools()],
            markdown=True,
            instructions=instructions,
            knowledge=knowledge_base,
            memory=AgentMemory(
                db=PgMemoryDb(
                    table_name="mlx_agent_memory",
                    db_url=memory_db_url
                ),
                create_session_summary=True,
            ),
            search_knowledge=True,
            read_chat_history=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            add_datetime_to_instructions=True,
        )

class ArchitectureAnalysisAgent(BaseMLXAgent):
    def __init__(self, knowledge_base, memory_db_url: str):
        super().__init__(
            description="""You are an expert in deep learning architectures, focusing on 
            understanding and analyzing model architectures, particularly text-to-video models.
            You have access to MLX and Wan2.1-T2V documentation through the knowledge base.""",
            instructions=[
                "ALWAYS start by searching the knowledge base for relevant MLX and Wan2.1 documentation",
                "Analyze model architecture and components",
                "Identify key neural network layers",
                "Document model parameters and configurations",
                "Map dependencies and data flows",
                "Note critical architectural features"
            ],
            knowledge_base=knowledge_base,
            memory_db_url=memory_db_url
        )

class MLXConversionAgent(BaseMLXAgent):
    def __init__(self, knowledge_base, memory_db_url: str):
        super().__init__(
            description="""You are an expert in MLX framework and model conversion.
            You have access to MLX documentation and implementation details through the knowledge base.""",
            instructions=[
                "ALWAYS search the knowledge base for MLX implementation patterns",
                "Identify MLX equivalent operations",
                "Plan tensor operation conversions",
                "Map PyTorch functions to MLX",
                "Consider Apple Silicon optimizations",
                "Document conversion requirements"
            ],
            knowledge_base=knowledge_base,
            memory_db_url=memory_db_url
        )

class DependencyAnalysisAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="""You are a dependency analysis expert who identifies and analyzes
            all required dependencies, libraries, and tools needed for the conversion process.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "List all required dependencies",
                "Check version compatibility",
                "Identify potential conflicts",
                "Suggest alternative packages if needed",
                "Document system requirements"
            ]
        )

class CodeConversionAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="""You are a code conversion specialist focused on translating
            PyTorch code to MLX, ensuring optimal performance and accuracy.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Convert PyTorch operations to MLX",
                "Optimize for Apple Silicon",
                "Maintain model accuracy",
                "Handle edge cases",
                "Document code changes"
            ]
        )

class TestingStrategyAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="""You are a testing specialist who designs and plans testing
            strategies for the converted model, ensuring functionality and performance.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Design test cases",
                "Plan validation strategies",
                "Define performance metrics",
                "Create comparison frameworks",
                "Document testing procedures"
            ]
        )

class OptimizationAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="""You are an optimization expert focused on improving the 
            performance of the converted model on Apple Silicon.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Identify optimization opportunities",
                "Suggest performance improvements",
                "Optimize memory usage",
                "Enhance computational efficiency",
                "Document optimization strategies"
            ]
        )

class DocumentationAgent:
    def __init__(self):
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4"),
            description="""You are a technical documentation specialist who creates
            comprehensive documentation for the conversion process and final implementation.""",
            tools=[ExaTools()],
            markdown=True,
            instructions=[
                "Create clear documentation",
                "Write usage guides",
                "Document known issues",
                "Provide examples",
                "Maintain conversion logs"
            ]
        ) 