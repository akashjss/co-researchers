import asyncio
from ai_co_scientist.coordinator import AICoScientist

async def main():
    scientist = AICoScientist()
    
    # Example research goal from the paper
    goal = """
    Find novel drug repurposing candidates for acute myeloid leukemia (AML).
    Focus on FDA-approved drugs that could be repurposed for AML treatment.
    Consider mechanisms of action, safety profiles, and potential combination therapies.
    """
    
    results = await scientist.research(goal)
    
    # Print the final report
    print("\nFinal Research Report:")
    print("=====================")
    print(results["report"])

if __name__ == "__main__":
    asyncio.run(main()) 