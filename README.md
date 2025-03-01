# AI Research & Development Agents

This repository contains three specialized AI agent applications built using Agno, Exa, and OpenAI:

1. AI Co-Scientist: A multi-agent system for scientific research and hypothesis generation
2. Deep Research: A comprehensive research system with multiple specialized agents
3. MLX T2V Researcher: A specialized system for converting text-to-video models to MLX framework

## 🚀 Features

### AI Co-Scientist
- Multi-agent system for scientific research
- Generates and evolves research hypotheses
- Performs simulated debates and rankings
- Specialized agents for different research phases:
  - Supervisor Agent: Orchestrates research process
  - Generation Agent: Creates initial hypotheses
  - Reflection Agent: Reviews and assesses hypotheses
  - Ranking Agent: Creates pairwise comparisons
  - Evolution Agent: Refines best hypotheses
  - Proximity Agent: Groups similar hypotheses
  - Meta-review Agent: Synthesizes insights

### Deep Research
- Comprehensive research system with specialized agents
- Performs deep analysis of any research topic
- Multiple verification and synthesis steps
- Specialized agents include:
  - Initial Research Agent: Creates research framework
  - Deep Dive Agent: Detailed investigation
  - Analysis Agent: Evaluates findings
  - Fact Check Agent: Verifies claims
  - Critical Review Agent: Challenges assumptions
  - Synthesis Agent: Integrates findings
  - Recommendation Agent: Provides actionable insights

### MLX T2V Researcher
- Specialized system for MLX model conversion
- Focuses on Apple Silicon optimization
- Comprehensive conversion planning and execution
- Specialized agents include:
  - Architecture Analysis Agent: Analyzes model structure
  - MLX Conversion Agent: Plans conversion process
  - Dependency Analysis Agent: Manages dependencies
  - Code Conversion Agent: Handles code translation
  - Testing Strategy Agent: Plans validation
  - Optimization Agent: Improves performance
  - Documentation Agent: Creates technical docs

## 📋 Requirements

```bash
# Core dependencies
pip install agno openai exa-py rich

# Environment variables
export OPENAI_API_KEY=your_openai_key
export EXA_API_KEY=your_exa_key
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-research-agents.git
cd ai-research-agents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### AI Co-Scientist

```python
from ai_co_scientist.coordinator import AICoScientist
import asyncio

async def main():
    scientist = AICoScientist()
    
    goal = """
    Find novel drug repurposing candidates for acute myeloid leukemia (AML).
    Focus on FDA-approved drugs that could be repurposed for AML treatment.
    """
    
    results = await scientist.research(goal)
    print(results["report"])

asyncio.run(main())
```

### Deep Research

```python
from deep_research.coordinator import DeepResearcher
import asyncio

async def main():
    researcher = DeepResearcher()
    
    topic = """
    What are the latest developments in quantum computing?
    Focus on recent breakthroughs and challenges.
    """
    
    results = await researcher.research(topic, depth="comprehensive")
    print(results["synthesis"])

asyncio.run(main())
```

### MLX T2V Researcher

```python
from mlx_t2v_researcher.coordinator import MLXConverter
import asyncio

async def main():
    converter = MLXConverter()
    
    model_path = "Wan-AI/Wan2.1-T2V-1.3B"
    results = await converter.plan_conversion(model_path)
    
    print(results["conversion_plan"])

asyncio.run(main())
```

## 📁 Project Structure

```
.
├── ai_co_scientist/
│   ├── __init__.py
│   ├── agents.py
│   └── coordinator.py
├── deep_research/
│   ├── __init__.py
│   ├── agents.py
│   └── coordinator.py
├── mlx_t2v_researcher/
│   ├── __init__.py
│   ├── agents.py
│   ├── coordinator.py
│   ├── docs/
│   │   ├── README.md
│   │   ├── mlx/
│   │   │   ├── core/
│   │   │   ├── examples/
│   │   │   └── api/
│   │   └── wan2.1/
│   │       ├── architecture/
│   │       ├── training/
│   │       └── inference/
│   ├── doc_loader.py
│   └── knowledge_base.py
├── examples/
│   ├── drug_repurposing.py
│   ├── research_example.py
│   └── convert_wan_t2v.py
├── README.md
└── requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Agno](https://github.com/agno-ai/agno) for the agent framework
- [Exa](https://exa.ai) for internet search capabilities
- [OpenAI](https://openai.com) for GPT-4 inference

## 🔗 Links

- Documentation: [Link to docs]
- Issue Tracker: [Link to issues]
- Project Homepage: [Link to homepage]

## 📧 Contact

For questions and support, please open an issue or contact [your contact info].

## Documentation Sources

The MLX T2V Researcher uses multiple documentation sources:

### Built-in Documentation
- Local documentation in `mlx_t2v_researcher/docs/`
- Automatically fetched source code and documentation from:
  - [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
  - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - High-performance Whisper model

### Loading Documentation
Documentation is automatically loaded when initializing the converter. You can also load custom documentation:

```python
from mlx_t2v_researcher.doc_loader import load_documentation

# Load both built-in and custom documentation
docs = load_documentation(custom_docs_path="path/to/your/docs")
``` 