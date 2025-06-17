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
    
    # Run single experiment with single dataset
    experiment_name = "two_step_complete_ft"  # You can change this to any experiment name from experiments.yaml
    dataset_name = "bitext"  # You can change this to any dataset name from experiments.yaml
    result = await runner.run_experiment(experiment_name, dataset_name)
    
    # Print results
    print("\nExperiment Results:")
    print(f"\nExperiment: {result['experiment']}")
    print(f"\nDataset: {result['dataset']}")
    print(f"\nMetrics: {result['metrics']}")
    print(f"\nintent_accuracy: {result['intent_accuracy']}")
    
    # Old code for running all experiments (commented out)
    # results = await runner.run_all_experiments()
    # print("\nExperiment Results:")
    # for result in results:
    #     print(f"\nExperiment: {result['experiment']}")
    #     print(f"Dataset: {result['dataset']}")
    #     print(f"Metrics: {result['metrics']}")

if __name__ == "__main__":
    asyncio.run(main())
