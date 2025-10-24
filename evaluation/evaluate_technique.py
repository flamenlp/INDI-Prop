import pandas as pd
import json
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

def evaluate_events(df):
    """
    Performs multi-label classification evaluation on the provided DataFrame.

    This function processes comma-separated labels, binarizes them, and
    generates a detailed classification report for different subsets of the data.

    Args:
        df (pd.DataFrame): A DataFrame that must contain 'Ground_Truth', 
                           'Predicted', and 'Event' columns.

    Returns:
        dict: A dictionary containing the classification reports for each unique 
              event and a combined overall report.
    """
    # --- 1. Data Pre-processing ---
    # Convert the comma-separated strings in the label columns into lists of strings.
    # This is necessary for the MultiLabelBinarizer to work correctly.
    try:
        df['Ground_Truth_list'] = df['Ground_Truth'].astype(str).str.split(',')
        df['Predicted_list'] = df['Predicted'].astype(str).str.split(',')
    except Exception as e:
        raise ValueError(f"Error splitting label columns. Ensure they are comma-separated strings. Details: {e}")

    # --- 2. Identify All Unique Labels ---
    # Create a comprehensive list of all possible labels present in the dataset.
    # This ensures the binary vectors are consistent across all evaluations.
    all_labels = sorted(list(set(
        label.strip() for sublist in df['Ground_Truth_list'] for label in sublist if label.strip()) |
        set(label.strip() for sublist in df['Predicted_list'] for label in sublist if label.strip())
    ))

    if not all_labels:
        print("Warning: No labels found in 'Ground_Truth' or 'Predicted' columns.")
        return {}

    # --- 3. Initialize the Binarizer ---
    # The MultiLabelBinarizer transforms lists of labels into a binary matrix format
    # (e.g., [G1, G4] becomes [1, 0, 0, 1, 0, 0, 0] if all labels are G1-G7).
    mlb = MultiLabelBinarizer(classes=all_labels)

    results = {}
    
    # --- 4. Evaluate Each Event Separately ---
    unique_events = df['Event'].unique()
    for event in unique_events:
        event_df = df[df['Event'] == event]
        
        # Transform the ground truth and predicted labels into binary format
        y_true = mlb.fit_transform(event_df['Ground_Truth_list'])
        y_pred = mlb.transform(event_df['Predicted_list'])
        
        # Generate the classification report and store it in the results dictionary
        # output_dict=True returns the report in a structured dictionary format.
        report = classification_report(
            y_true,
            y_pred,
            target_names=all_labels,
            zero_division=0,
            output_dict=True
        )
        results[event] = report
        
    # --- 5. Perform a Combined Overall Evaluation ---
    # This runs the same evaluation on the entire dataset.
    if len(unique_events) > 1:
        y_true_combined = mlb.fit_transform(df['Ground_Truth_list'])
        y_pred_combined = mlb.transform(df['Predicted_list'])
        
        report_combined = classification_report(
            y_true_combined,
            y_pred_combined,
            target_names=all_labels,
            zero_division=0,
            output_dict=True
        )
        results["Combined_Overall"] = report_combined
        
    return results

def main():
    """
    Main function to handle command-line arguments, read the CSV,
    run the evaluation, and handle the output.
    """
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate multi-label classification results from a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file.\n"
             "The file must contain the following columns:\n"
             "  - Ground_Truth: Comma-separated ground truth labels.\n"
             "  - Predicted: Comma-separated predicted labels.\n"
             "  - Event: The event category for the row."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_json",
        help="Optional: Path to save the evaluation results as a JSON file."
    )
    args = parser.parse_args()

    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(args.input_csv)
        
        # Validate that the necessary columns exist in the CSV
        required_columns = ['Ground_Truth', 'Predicted', 'Event']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input CSV must contain the columns: {required_columns}")
            return

        # Run the core evaluation function
        evaluation_results = evaluate_events(df)

        # Handle the output: either print to console or save to a file
        output_str = json.dumps(evaluation_results, indent=4)

        if args.output_json:
            with open(args.output_json, 'w') as f:
                f.write(output_str)
            print(f"Evaluation results successfully saved to {args.output_json}")
        else:
            print(output_str)

    except FileNotFoundError:
        print(f"Error: The file '{args.input_csv}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.input_csv}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
