import asyncio
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from .coordinator import MLXConverter

class DocumentationLoader:
    def __init__(self, base_path: Path = Path(__file__).parent / "docs"):
        self.base_path = base_path
        self.mlx_path = base_path / "mlx"
        self.wan_path = base_path / "wan2.1"
        self.console = Console()

    async def load_documentation(self, converter: MLXConverter) -> None:
        """Load documentation into the knowledge base"""
        
        # Load MLX documentation with structure preservation
        mlx_docs = self._load_docs_with_structure(self.mlx_path, "MLX")
        
        # Load Wan2.1 documentation with structure preservation
        wan_docs = self._load_docs_with_structure(self.wan_path, "Wan2.1")
        
        # Combine and format documentation
        formatted_docs = self._format_documentation(mlx_docs, wan_docs)
        
        # Load into knowledge base
        await converter.load_documentation(
            mlx_docs=formatted_docs["mlx"],
            wan_docs=formatted_docs["wan"]
        )

    def _load_docs_with_structure(self, path: Path, doc_type: str) -> List[Dict[str, str]]:
        """Load documentation while preserving directory structure"""
        docs = []
        if not path.exists():
            self.console.print(f"[yellow]Warning: {doc_type} documentation path does not exist: {path}[/yellow]")
            return docs

        for doc_file in path.rglob("*.md"):
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    relative_path = doc_file.relative_to(path)
                    section = str(relative_path.parent)
                    docs.append({
                        "content": f.read(),
                        "path": str(relative_path),
                        "section": section if section != "." else "root"
                    })
            except Exception as e:
                self.console.print(f"[red]Error loading {doc_file}: {str(e)}[/red]")
        
        return docs

    def _format_documentation(
        self, 
        mlx_docs: List[Dict[str, str]], 
        wan_docs: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Format documentation with section headers and metadata"""
        
        def format_section(docs: List[Dict[str, str]], doc_type: str) -> str:
            formatted = f"# {doc_type} Documentation\n\n"
            
            # Group by section
            sections = {}
            for doc in docs:
                section = doc["section"]
                if section not in sections:
                    sections[section] = []
                sections[section].append(doc)
            
            # Format each section
            for section, section_docs in sections.items():
                formatted += f"## {section}\n\n"
                for doc in section_docs:
                    formatted += f"### File: {doc['path']}\n\n{doc['content']}\n\n"
            
            return formatted

        return {
            "mlx": format_section(mlx_docs, "MLX"),
            "wan": format_section(wan_docs, "Wan2.1-T2V")
        }

async def load_documentation(
    docs_path: Path = None,
    converter: MLXConverter = None
) -> None:
    """Convenience function to load documentation"""
    loader = DocumentationLoader(docs_path)
    await loader.load_documentation(converter) 