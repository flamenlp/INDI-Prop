import pandas as pd
import numpy as np
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any


def evaluate_bias_classification(csv_path: str, output_json: str = None) -> Dict[str, Any]:
    """
    Evaluates bias classification performance using ground truth and predictions.
    
    Args:
        csv_path: Path to CSV file containing 'Bias' and 'Detected Bias Label' columns
        output_json: Optional path to save metrics as JSON
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Validate required columns exist
    if 'Bias' not in df.columns:
        raise ValueError("CSV file must contain 'Bias' column for ground truth")
    
    # Check for prediction column (handle different naming variations)
    pred_column = None
    for col in ['Detected Bias Label', 'Detected_Bias_Label', 'Detected Bias']:
        if col in df.columns:
            pred_column = col
            break
    
    if pred_column is None:
        raise ValueError("CSV file must contain one of: 'Detected Bias Label', 'Detected_Bias_Label', or 'Detected Bias' column for predictions")
    
    # Extract ground truth and predictions
    y_true = df['Bias'].values
    y_pred = df[pred_column].values
    
    # Remove rows with missing values
    valid_mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    # Convert to lowercase for consistency
    y_true = np.array([str(label).lower() for label in y_true])
    y_pred = np.array([str(label).lower() for label in y_pred])
    
    # Define label order (alphabetical for consistency)
    labels = ['left', 'neutral', 'right']
    
    # Calculate metrics
    print("\n" + "="*80)
    print("BIAS CLASSIFICATION EVALUATION")
    print("="*80)
    print(f"\nInput file: {csv_path}")
    print(f"Total samples: {len(y_true)}")
    print(f"Ground truth column: 'Bias'")
    print(f"Prediction column: '{pred_column}'")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification report with different averaging methods
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    report = classification_report(y_true, y_pred, labels=labels, 
                                   target_names=labels, digits=4, zero_division=0)
    print(report)
    
    # Get detailed metrics for each averaging method
    from sklearn.metrics import precision_recall_fscore_support
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', labels=labels, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=labels, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=labels, zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("\n" + "-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    print(f"\n{'':>10} {'Predicted':^40}")
    print(f"{'Actual':>10} {' | '.join([f'{label:>12}' for label in labels])}")
    print("-"*80)
    for i, label in enumerate(labels):
        print(f"{label:>10} {' | '.join([f'{cm[i][j]:>12}' for j in range(len(labels))])}")
    
    # Summary of averaging methods
    print("\n" + "-"*80)
    print("AVERAGING METHODS SUMMARY")
    print("-"*80)
    print(f"\n{'Metric':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-"*80)
    print(f"{'Micro Avg':<15} {precision_micro:>12.4f} {recall_micro:>12.4f} {f1_micro:>12.4f}")
    print(f"{'Macro Avg':<15} {precision_macro:>12.4f} {recall_macro:>12.4f} {f1_macro:>12.4f}")
    print(f"{'Weighted Avg':<15} {precision_weighted:>12.4f} {recall_weighted:>12.4f} {f1_weighted:>12.4f}")
    print("-"*80)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'micro_avg': {
            'precision': float(precision_micro),
            'recall': float(recall_micro),
            'f1_score': float(f1_micro)
        },
        'macro_avg': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1_score': float(f1_macro)
        },
        'weighted_avg': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1_score': float(f1_weighted)
        },
        'per_class': {
            labels[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(len(labels))
        },
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'total_samples': int(len(y_true))
    }
    
    # Save to JSON if output path provided
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {output_json}")
    
    print("\n" + "="*80 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate bias classification performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_bias.py predictions.csv
  
  # Evaluate and save metrics to JSON
  python evaluate_bias.py predictions.csv -o metrics.json
  python evaluate_bias.py predictions.csv --output metrics.json
        """
    )
    
    parser.add_argument('input_csv', 
                       help='Path to CSV file with Bias and Detected Bias Label columns')
    parser.add_argument('-o', '--output', 
                       help='Optional: Path to save metrics as JSON file')
    
    args = parser.parse_args()
    
    try:
        evaluate_bias_classification(args.input_csv, args.output)
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
