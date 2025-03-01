import asyncio
from pathlib import Path
from mlx_t2v_researcher.coordinator import MLXConverter
from mlx_t2v_researcher.doc_loader import load_documentation

async def main():
    # Initialize the converter
    converter = MLXConverter()
    
    # Load documentation including MLX and Wan2.1 repositories
    docs = load_documentation()
    print(f"Loaded {len(docs)} documentation files")
    
    # Specify the model you want to convert
    model_path = "Wan-AI/Wan2.1-T2V-1.3B"
    
    # Get the conversion plan
    results = await converter.plan_conversion(model_path)
    
    # Print the conversion plan
    print("\nConversion Plan:")
    print(results["conversion_plan"])

if __name__ == "__main__":
    asyncio.run(main()) 