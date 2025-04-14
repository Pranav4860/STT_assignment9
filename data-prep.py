from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlrun

@mlrun.handler(outputs=["breast_cancer_data", "label_column"])
def prep_data(context,format="csv"):
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df['target'] = data.target
    
    context.logger.info('saving cancer dataset to {}'.format(context.artifact_path))
    context.log_dataset("breast_cancer_data", df=df, format="csv", index=False)
    
    return df, "target"

if __name__ == "__main__":
    with mlrun.get_or_create_ctx("breast-cancer", upload_artifacts=True) as context:
        prep_data(context, context.get_param("format", "csv"))