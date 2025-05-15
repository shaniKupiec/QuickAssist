"""Main entry point for running experiments."""

import os
import sys
import asyncio
from components.experiment_runner import ExperimentRunner

def setup_environment():
    """Setup the environment for running experiments."""
    # Add the project root to Python path to make imports work
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)

    # Load environment variables if needed
    from dotenv import load_dotenv
    load_dotenv()

async def main():
    """Main function to run experiments."""
    # Setup environment
    setup_environment()
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Run all experiments
    results = await runner.run_all_experiments()
    
    # TODO: Save results to file or display them
    print("\nExperiment Results:")
    for result in results:
        print(f"\nExperiment: {result['experiment']}")
        print(f"Dataset: {result['dataset']}")
        print(f"Metrics: {result['metrics']}")

if __name__ == "__main__":
    asyncio.run(main())
