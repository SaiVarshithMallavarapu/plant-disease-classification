# main.py
import argparse
import sys
import os

# Add src directory to Python path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Now import from src
try:
    from src import train
    from src import evaluate
    from src import config # Needed to potentially access config defaults or paths
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure you are running this script from the 'soybean_disease_classifier' directory")
    print("or that the 'src' directory is correctly in your Python path.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate DenseNet models for Soybean Disease Classification.")

    parser.add_argument('action', choices=['train', 'evaluate'],
                        help="Action to perform: 'train' or 'evaluate'.")
    parser.add_argument('--model', choices=['DenseNet121', 'DenseNet201'], required=True,
                        help="Specify the DenseNet model to use: 'DenseNet121' or 'DenseNet201'.")
    # Add more arguments if needed (e.g., override batch size, epochs, data dir)
    # parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Path to the main data directory.')
    # parser.add_argument('--epochs', type=int, help='Override total number of training epochs.')

    args = parser.parse_args()

    model_choice = args.model
    action = args.action

    print(f"Selected Action: {action}")
    print(f"Selected Model: {model_choice}")

    # # --- (Optional) Override config values based on args ---
    # if args.data_dir:
    #     # Need to update all derived paths in config if DATA_DIR changes
    #     print(f"Overriding DATA_DIR to: {args.data_dir}")
    #     config.DATA_DIR = args.data_dir
    #     config.TRAIN_DIR = os.path.join(config.DATA_DIR, 'train')
    #     config.VAL_DIR = os.path.join(config.DATA_DIR, 'validation')
    #     config.TEST_DIR = os.path.join(config.DATA_DIR, 'test')
    #
    # if args.epochs:
    #     print(f"Overriding TOTAL_EPOCHS to: {args.epochs}")
    #     # Be careful how you adjust INITIAL vs FINE_TUNE epochs if overriding total
    #     config.TOTAL_EPOCHS = args.epochs
    #     # A simple approach: scale initial/fine-tune proportionally, or set one fixed
    #     # This needs careful consideration based on your training strategy.
    #     # Example: keep INITIAL_EPOCHS fixed, adjust FINE_TUNE_EPOCHS
    #     if config.TOTAL_EPOCHS > config.INITIAL_EPOCHS:
    #         config.FINE_TUNE_EPOCHS = config.TOTAL_EPOCHS - config.INITIAL_EPOCHS
    #     else: # If total is less than initial, maybe only do initial training?
    #         config.INITIAL_EPOCHS = config.TOTAL_EPOCHS
    #         config.FINE_TUNE_EPOCHS = 0
    #     print(f"Adjusted INITIAL_EPOCHS: {config.INITIAL_EPOCHS}, FINE_TUNE_EPOCHS: {config.FINE_TUNE_EPOCHS}")


    try:
        if action == 'train':
            train.train_model(model_choice)
        elif action == 'evaluate':
            evaluate.evaluate_trained_model(model_choice)
    except Exception as e:
        print(f"\n--- An error occurred during execution ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Script execution finished ---")

if __name__ == "__main__":
    main()