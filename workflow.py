import mlrun
from kfp import dsl

@dsl.pipeline(name="mlrun_pipeline")
def pipeline(model_name="breast-cancer"):
    
    ingest = mlrun.run_function(
        "breast_cancer_data",
        name="breast_cancer_data",
        params={"format": "pq", "model_name": model_name},
        outputs=["breast_cancer_data"],
    )
    
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": ingest.outputs["breast_cancer_data"]},
        hyperparams={
        "n_estimators": [10, 100,200],
            "max_depth": [2, 5, 10]
        },
        selector="max.accuracy",
        outputs=["model"],
    )
    
    deploy = mlrun.deploy_function(
        "serving",
        models=[{"key": model_name, "model_path": train.outputs["model"], "class_name": "ClassifierModel"}],
        mock=True
    )    