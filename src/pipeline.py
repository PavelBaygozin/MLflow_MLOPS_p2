from clearml import PipelineDecorator, Task
import subprocess
import time

@PipelineDecorator.component(cache=False, execution_queue="default")
def dvc_pull():
    try:
        print("Starting dvc_pull: downloading data from the remote DVC storage...")
        subprocess.run(["dvc", "pull"], check=True)
        print("dvc_pull completed successfully: data downloaded.")
        return Task.current_task().id  # Return task_id for use in process_data
    except subprocess.CalledProcessError as e:
        print(f"Error in dvc_pull: {e}")
        raise

@PipelineDecorator.component(cache=False, execution_queue="default", parents=["dvc_pull"])
def process_data(dvc_pull_task_id):
    try:
        print("Starting process_data: processing data...")

        # Wait for dvc_pull to complete
        dvc_pull_task = Task.get_task(task_id=dvc_pull_task_id)
        while not dvc_pull_task.completed:
            time.sleep(5)  # Wait 5 seconds before checking again

        # Run data processing
        subprocess.run(["python", "src/main_data.py"], check=True)
        print("process_data completed successfully: data processed.")
        return Task.current_task().id  # Return task_id for use in train_model
    except subprocess.CalledProcessError as e:
        print(f"Error in process_data: {e}")
        raise

@PipelineDecorator.component(execution_queue="default", parents=["process_data"])
def train_model(process_data_task_id, param):
    try:
        print(f"Starting train_model with param={param}: training the model...")

        # Wait for process_data to complete
        process_data_task = Task.get_task(task_id=process_data_task_id)
        while not process_data_task.completed:
            time.sleep(5)  # Wait 5 seconds before checking again

        # Run model training with the specified param
        print(f"Running train_data.py with n_estimators={param}")
        subprocess.run(["python", "src/train_data.py", "--n_estimators", str(param)], check=True)
        print(f"train_model with param={param} completed successfully: model trained.")
        return Task.current_task().id  # Return task_id for use in dvc_repro
    except subprocess.CalledProcessError as e:
        print(f"Error in train_model: {e}")
        print(f"Command failed: {e.cmd}")
        print(f"Output: {e.output}")
        raise

@PipelineDecorator.component(cache=False, execution_queue="default", parents=["train_model"])
def dvc_repro(train_model_task_id):
    try:
        print("Starting dvc_repro: reproducing the DVC pipeline...")

        # Wait for train_model to complete
        train_model_task = Task.get_task(task_id=train_model_task_id)
        while not train_model_task.completed:
            time.sleep(5)  # Wait 5 seconds before checking again

        # Run dvc repro to update data
        subprocess.run(["dvc", "repro"], check=True)
        print("dvc_repro completed successfully: DVC pipeline reproduced.")
        return Task.current_task().id  # Return task_id for use in dvc_push
    except subprocess.CalledProcessError as e:
        print(f"Error in dvc_repro: {e}")
        raise

@PipelineDecorator.component(cache=False, execution_queue="default", parents=["dvc_repro"])
def dvc_push(dvc_repro_task_id):
    try:
        print("Starting dvc_push: uploading data to the remote DVC storage...")

        # Wait for dvc_repro to complete
        dvc_repro_task = Task.get_task(task_id=dvc_repro_task_id)
        while not dvc_repro_task.completed:
            time.sleep(5)  # Wait 5 seconds before checking again

        # Upload data to the remote DVC storage
        subprocess.run(["dvc", "push"], check=True)
        print("dvc_push completed successfully: data uploaded to the remote storage.")
    except subprocess.CalledProcessError as e:
        print(f"Error in dvc_push: {e}")
        raise

@PipelineDecorator.pipeline(
    name='random_forest_pipeline',
    project='RandomForestExperiment',
    version='0.1'
)
def text_classification_pipeline_logic():
    dvc_pull_task_id = dvc_pull()
    process_data_task_id = process_data(dvc_pull_task_id)

    # Run models with different param sizes sequentially
    train_model_task_id_50 = train_model(process_data_task_id, param=50)
    train_model_task_id_100 = train_model(train_model_task_id_50, param=100)  # Depends on the completion of the first model

    # Update DVC and upload data
    dvc_repro_task_id = dvc_repro(train_model_task_id_50)
    dvc_push(dvc_repro_task_id)

if __name__ == '__main__':
    # Run the pipeline locally (for debugging)
    PipelineDecorator.run_locally()
    text_classification_pipeline_logic()