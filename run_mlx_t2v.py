import asyncio
from mlx_t2v_researcher.agents import create_base_agent

async def main():
    # Initialize agents
    architecture_analyzer = create_base_agent(
        name="Architecture Analyzer",
        system_prompt="""You are an expert in ML model architectures, specializing in converting models to MLX.
        Analyze model architectures and identify key components that need conversion."""
    )
    
    mlx_converter = create_base_agent(
        name="MLX Converter",
        system_prompt="""You are an expert in MLX framework and model conversion.
        Create detailed plans for converting models to MLX, considering Apple Silicon optimizations."""
    )
    
    code_converter = create_base_agent(
        name="Code Converter",
        system_prompt="""You are an expert in translating ML model code to MLX.
        Focus on efficient and optimized implementations for Apple Silicon."""
    )

    # Specify the model to convert
    model_path = "Wan-AI/Wan2.1-T2V-1.3B"
    
    # Get analysis and plans
    print("\nAnalyzing model architecture...")
    arch_response = await architecture_analyzer.arun(
        f"Analyze the architecture of {model_path} for MLX conversion."
    )
    architecture_analysis = arch_response.content if hasattr(arch_response, 'content') else str(arch_response)

    print("\nCreating MLX conversion plan...")
    plan_response = await mlx_converter.arun(
        f"Create MLX conversion plan for {model_path}. Analysis: {architecture_analysis[:1000]}"
    )
    conversion_plan = plan_response.content if hasattr(plan_response, 'content') else str(plan_response)

    print("\nDeveloping code conversion strategy...")
    code_response = await code_converter.arun(
        f"Create code strategy. Plan: {conversion_plan[:1000]}"
    )
    code_strategy = code_response.content if hasattr(code_response, 'content') else str(code_response)

    # Print results
    print("\nArchitecture Analysis:")
    print(architecture_analysis)
    
    print("\nConversion Plan:")
    print(conversion_plan)
    
    print("\nCode Strategy:")
    print(code_strategy)

if __name__ == "__main__":
    asyncio.run(main()) 