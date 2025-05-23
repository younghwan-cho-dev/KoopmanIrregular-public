import argparse
from src.experiment_runners import robustness_test, discover_high_order

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Koopman experiments.")
    
    # Enforce the task parameter.
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["robustness", "high_order"],
        required=True,
        help="Specify which experiment to run."
    )

    args = parser.parse_args()

    if args.task == "robustness":
        print("Start robustness test...")
        robustness_test()
    elif args.task == "high_order":
        print("Start high-order identification test...")
        discover_high_order()
    print("Ending experiment...")