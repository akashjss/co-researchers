# MLX T2V Researcher

A specialized AI agent system for converting text-to-video models to the MLX framework, optimized for Apple Silicon.

## Features

- Specialized AI agents for model conversion:
  - Architecture Analysis Agent: Analyzes model structure
  - MLX Conversion Agent: Plans conversion process
  - Code Conversion Agent: Handles code translation
- Uses Exa for real-time documentation search
- Optimized for Apple Silicon
- Provides detailed conversion plans and strategies

## Requirements

- Python 3.8+
- OpenAI API key
- Exa API key

## Quick Start

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_key
EXA_API_KEY=your_exa_key
```

4. Run the application:
```bash
python run_mlx_t2v.py
```

## Project Structure

```
.
├── mlx_t2v_researcher/
│   ├── __init__.py
│   └── agents.py
├── run_mlx_t2v.py
├── setup.py
└── README.md
```

## Usage Example

```python
import asyncio
from mlx_t2v_researcher.agents import create_base_agent

async def main():
    # Initialize an agent
    converter = create_base_agent(
        name="MLX Converter",
        system_prompt="Create MLX conversion plans optimized for Apple Silicon"
    )
    
    # Get conversion plan
    response = await converter.arun(
        "Create a conversion plan for Wan-AI/Wan2.1-T2V-1.3B"
    )
    print(response)

asyncio.run(main())
```

## 🙏 Acknowledgments

- [Agno](https://github.com/agno-ai/agno) for the agent framework
- [Exa](https://exa.ai) for internet search capabilities
- [OpenAI](https://openai.com) for GPT-4 inference
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 