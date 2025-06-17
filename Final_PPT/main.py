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

    from dotenv import load_dotenv
    load_dotenv()

async def main():
    """Main function to run experiments."""
    setup_environment()
    
    runner = ExperimentRunner()
    
    experiment_name = "two_step_complete_ft"
    dataset_name = "bitext"
    result = await runner.run_experiment(experiment_name, dataset_name)
    
    # Print results
    print("\nExperiment Results:")
    print(f"\nExperiment: {result['experiment']}")
    print(f"\nDataset: {result['dataset']}")
    print(f"\nMetrics: {result['metrics']}")
    print(f"\nIntent_accuracy: {result['intent_accuracy']}")
    

if __name__ == "__main__":
    asyncio.run(main())
