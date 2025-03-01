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
    console.print("\n[bold green]Initial Research Framework[/bold green]")
    console.print(Markdown(results["framework"]))
    
    console.print("\n[bold cyan]Deep Dive Research[/bold cyan]")
    console.print(Markdown(results["deep_dive"]))
    
    console.print("\n[bold yellow]Analysis & Fact Check[/bold yellow]")
    console.print(Markdown(results["analysis"]))
    console.print(Markdown(results["fact_check"]))
    
    console.print("\n[bold red]Critical Review[/bold red]")
    console.print(Markdown(results["critique"]))
    
    console.print("\n[bold magenta]Final Synthesis[/bold magenta]")
    console.print(Markdown(results["synthesis"]))
    
    console.print("\n[bold green]Recommendations[/bold green]")
    console.print(Markdown(results["recommendations"]))

if __name__ == "__main__":
    asyncio.run(main()) 