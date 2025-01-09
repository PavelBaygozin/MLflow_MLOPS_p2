from clearml import Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import numpy as np
import argparse  # Import the argparse module

# Limit the number of CPU threads
torch.set_num_threads(2)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RandomForestClassifier with n_estimators as argument")
    parser.add_argument('--n_estimators', type=int, nargs='+', required=True,
                        help="List of n_estimators values to experiment with")
    args = parser.parse_args()

    # Initialize ClearML task
    task = Task.init(project_name="MyProject", task_name="RandomForestExperiment")

    # Get the absolute path to the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load preprocessed data
    processed_data_dir = os.path.join(script_dir, '../data/output')
    train_data_path = os.path.join(processed_data_dir, 'train_data.pt')
    test_data_path = os.path.join(processed_data_dir, 'test_data.pt')

    train_data = torch.load(train_data_path, weights_only=False)
    test_data = torch.load(test_data_path, weights_only=False)

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    # Prepare data for RandomForest
    def prepare_data(dataset):
        features = dataset.tensors[0].numpy()  # Convert tensor to NumPy array
        labels = dataset.tensors[1].numpy()
        return features, labels

    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Experiment with different n_estimators values
    n_estimators_list = args.n_estimators  # Use values from command-line arguments

    for n_estimators in n_estimators_list:
        print(f"\nRunning experiment with n_estimators = {n_estimators}")

        # Create and train the RandomForest model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=2)  # n_jobs to use 2 threads
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")

        # Log parameters and metrics in ClearML
        task.get_logger().report_scalar(title="Metrics", series="Accuracy", value=accuracy, iteration=n_estimators)
        task.get_logger().report_scalar(title="Metrics", series="F1 Score", value=f1, iteration=n_estimators)
        task.get_logger().report_single_value(name="n_estimators", value=n_estimators)

        # Save the model
        model_path = f"model_n_estimators_{n_estimators}.pkl"
        import joblib
        joblib.dump(model, model_path)
        task.upload_artifact(name=model_path, artifact_object=model_path)

if __name__ == "__main__":
    main()