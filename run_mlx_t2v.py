import asyncio
from mlx_t2v_researcher.coordinator import MLXConverter

async def main():
    # Initialize the converter
    converter = MLXConverter()
    
    # Specify the model you want to convert
    model_path = "Wan-AI/Wan2.1-T2V-1.3B"
    
    # Get the conversion plan
    results = await converter.plan_conversion(model_path)
    
    # Print the results
    print("\nArchitecture Analysis:")
    print(results["architecture_analysis"])
    
    print("\nConversion Plan:")
    print(results["conversion_plan"])
    
    print("\nCode Strategy:")
    print(results["code_strategy"])

if __name__ == "__main__":
    asyncio.run(main()) 