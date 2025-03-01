import asyncio
from deep_research.coordinator import DeepResearcher
from rich.console import Console
from rich.markdown import Markdown

async def main():
    console = Console()
    researcher = DeepResearcher()
    
    # Example research topic
    topic = """
    What are the latest developments and challenges in quantum computing?
    Focus on:
    - Recent breakthroughs
    - Current limitations
    - Potential applications
    - Major players in the field
    """
    
    console.print("\n[bold blue]Starting Deep Research...[/bold blue]")
    console.print("[dim]This may take a few minutes...[/dim]\n")
    
    results = await researcher.research(topic, depth="comprehensive")
    
    # Print results in a nicely formatted way
    console.print("\n[bold green]Initial Research Findings[/bold green]")
    console.print(Markdown(results["research"]))
    
    console.print("\n[bold yellow]Fact Check Results[/bold yellow]")
    console.print(Markdown(results["fact_check"]))
    
    console.print("\n[bold magenta]Final Synthesis[/bold magenta]")
    console.print(Markdown(results["synthesis"]))

if __name__ == "__main__":
    asyncio.run(main()) 