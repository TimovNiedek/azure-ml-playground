import time
import uuid

from azure.ai.ml import Input, MLClient, Output, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import PipelineJob
from azure.identity import DefaultAzureCredential

from f1_data_predictions.utils import get_ml_client

ml_client = get_ml_client()

registry_name = "azureml"
ml_client_registry = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=ml_client.subscription_id,
    resource_group_name=ml_client.resource_group_name,
    registry_name=registry_name,
)

label = "latest"

rai_constructor_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_constructor", label=label
)

# we get latest version and use the same version for all components
version = rai_constructor_component.version
print("The current version of RAI built-in components is: " + version)

rai_erroranalysis_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_erroranalysis", version=version
)

rai_explanation_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_explanation", version=version
)

rai_gather_component = ml_client_registry.components.get(
    name="microsoft_azureml_rai_tabular_insight_gather", version=version
)


@dsl.pipeline(
    compute="cpu-cluster",
    description="RAI insights on diabetes data",
    experiment_name="RAI_insights_f1",
)
def rai_decision_pipeline(target_column_name, train_data, test_data):
    model = ml_client.models.get("f1-model", "2")
    expected_model_id = f"{model.name}:{model.version}"
    print(expected_model_id)

    # Initiate the RAIInsights
    create_rai_job = rai_constructor_component(
        title="RAI dashboard f1",
        task_type="classification",
        model_info=expected_model_id,
        model_input=Input(
            type=AssetTypes.MLFLOW_MODEL, path=f"azureml:{expected_model_id}"
        ),
        train_dataset=train_data,
        test_dataset=test_data,
        target_column_name=target_column_name,
    )
    create_rai_job.set_limits(timeout=300)

    # Add error analysis
    error_job = rai_erroranalysis_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
    )
    error_job.set_limits(timeout=300)

    # Add explanations
    explanation_job = rai_explanation_component(
        rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        comment="add explanation",
    )
    explanation_job.set_limits(timeout=300)

    # Combine everything
    rai_gather_job = rai_gather_component(
        constructor=create_rai_job.outputs.rai_insights_dashboard,
        insight_3=error_job.outputs.error_analysis,
        insight_4=explanation_job.outputs.explanation,
    )
    rai_gather_job.set_limits(timeout=300)

    rai_gather_job.outputs.dashboard.mode = "upload"

    return {
        "dashboard": rai_gather_job.outputs.dashboard,
    }


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    print(
        f"Pipeline job can be accessed in the following URL: {created_job.studio_url}"
    )

    while created_job.status not in [
        "Completed",
        "Failed",
        "Canceled",
        "NotResponding",
    ]:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))
    assert created_job.status == "Completed"
    return created_job


def run_rai(
    train_data="azureml:train_data_parquet:1",
    test_data="azureml:test_data_parquet:1",
) -> PipelineJob:
    f1_train_pq = Input(
        type="mltable",
        path=train_data,
        mode="download",
    )
    f1_test_pq = Input(
        type="mltable",
        path=test_data,
        mode="download",
    )

    insights_pipeline_job = rai_decision_pipeline(
        target_column_name="pointsFinish",
        train_data=f1_train_pq,
        test_data=f1_test_pq,
    )

    # Workaround to enable the download
    rand_path = str(uuid.uuid4())
    insights_pipeline_job.outputs.dashboard = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/{rand_path}/dashboard/",
        mode="upload",
        type="uri_folder",
    )
    print("Dashboard can be found at:")
    print(insights_pipeline_job.outputs.dashboard.path)

    insights_job = submit_and_wait(ml_client, insights_pipeline_job)
    return insights_job
