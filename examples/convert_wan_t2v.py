import asyncio
from pathlib import Path
from mlx_t2v_researcher.coordinator import MLXConverter
from mlx_t2v_researcher.doc_loader import load_documentation
from rich.console import Console
from rich.markdown import Markdown

async def main():
    console = Console()
    converter = MLXConverter()
    
    # Load documentation from default path
    await load_documentation(converter=converter)
    
    # Or specify custom documentation path:
    # await load_documentation(
    #     docs_path=Path("custom/docs/path"),
    #     converter=converter
    # )
    
    model_path = "Wan-AI/Wan2.1-T2V-1.3B"
    
    console.print("\n[bold blue]Starting MLX Conversion Planning...[/bold blue]")
    console.print("[dim]This may take a few minutes...[/dim]\n")
    
    results = await converter.plan_conversion(model_path)
    
    # Print results in a nicely formatted way
    sections = [
        ("Architecture Analysis", "architecture_analysis"),
        ("Dependencies", "dependencies"),
        ("Conversion Plan", "conversion_plan"),
        ("Code Strategy", "code_strategy"),
        ("Testing Plan", "testing_plan"),
        ("Optimization Plan", "optimization_plan"),
        ("Documentation", "documentation")
    ]
    
    for title, key in sections:
        console.print(f"\n[bold green]{title}[/bold green]")
        console.print(Markdown(results[key]))
        console.print("\n" + "-"*50)

if __name__ == "__main__":
    asyncio.run(main()) 