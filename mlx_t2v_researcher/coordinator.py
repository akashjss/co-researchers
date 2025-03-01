from typing import Dict, Any
from rich.console import Console
from dotenv import load_dotenv
import os
from .agents import create_base_agent

# Load environment variables
load_dotenv()

class MLXConverter:
    def __init__(self):
        self.console = Console()
        
        # Initialize specialized agents
        self.architecture_analyzer = create_base_agent(
            name="Architecture Analyzer",
            system_prompt="""You are an expert in ML model architectures, specializing in converting models to MLX.
            Analyze model architectures and identify key components that need conversion."""
        )
        
        self.mlx_converter = create_base_agent(
            name="MLX Converter",
            system_prompt="""You are an expert in MLX framework and model conversion.
            Create detailed plans for converting models to MLX, considering Apple Silicon optimizations."""
        )
        
        self.code_converter = create_base_agent(
            name="Code Converter",
            system_prompt="""You are an expert in translating ML model code to MLX.
            Focus on efficient and optimized implementations for Apple Silicon."""
        )

    async def plan_conversion(self, model_path: str) -> Dict[str, Any]:
        """Plan and execute the conversion of the model to MLX"""
        try:
            self.console.print(f"\n[bold]Starting conversion planning for {model_path}[/bold]")

            # 1. Analyze model architecture
            self.console.print("\n[cyan]Analyzing model architecture...[/cyan]")
            arch_response = await self.architecture_analyzer.arun(
                f"Analyze the architecture of {model_path} for MLX conversion."
            )
            architecture_analysis = arch_response.content if hasattr(arch_response, 'content') else str(arch_response)

            # 2. Plan MLX conversion
            self.console.print("\n[cyan]Creating MLX conversion plan...[/cyan]")
            plan_response = await self.mlx_converter.arun(
                f"Create MLX conversion plan for {model_path}. Analysis: {architecture_analysis[:1000]}"
            )
            conversion_plan = plan_response.content if hasattr(plan_response, 'content') else str(plan_response)

            # 3. Generate code conversion strategy
            self.console.print("\n[cyan]Developing code conversion strategy...[/cyan]")
            code_response = await self.code_converter.arun(
                f"Create code strategy. Plan: {conversion_plan[:1000]}"
            )
            code_strategy = code_response.content if hasattr(code_response, 'content') else str(code_response)

            return {
                "architecture_analysis": architecture_analysis,
                "conversion_plan": conversion_plan,
                "code_strategy": code_strategy
            }
        except Exception as e:
            self.console.print(f"[red]Error during conversion planning: {str(e)}[/red]")
            raise 