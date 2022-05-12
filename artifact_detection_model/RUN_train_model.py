import json

from sklearn.svm import LinearSVC

from artifact_detection_model.dataset_creation import get_training_and_test_set, \
    get_manual_validation_data_set
from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import PROJECT_ROOT, REVIEWER_1_PATH, REVIEWER_2_PATH
from evaluation.utils import nlon_performance, validation_performance_on_dataset

log = Logger()

OUT_PATH = PROJECT_ROOT + 'artifact_detection_model/out/'


def main():
    r1data, r1target = get_manual_validation_data_set(REVIEWER_1_PATH)
    r2data, r2target = get_manual_validation_data_set(REVIEWER_2_PATH)

    train_frac = 0.4
    df_train, df_test = get_training_and_test_set()
    df_train = df_train.sample(frac=train_frac, random_state=42)
    report, pipeline = run_ml_artifact_training(df_train, df_test, LinearSVC(random_state=42), model_output_path=OUT_PATH)

    report.update({'name': 'LSVCdef'})
    report.update({'train_frac': train_frac})

    report.update(validation_performance_on_dataset(pipeline, r1data, r1target, 'reviewer_1'))
    report.update(validation_performance_on_dataset(pipeline, r2data, r2target, 'reviewer_2', output_misclassified=OUT_PATH))
    report.update(nlon_performance(pipeline, 'Fabio'))
    report.update(nlon_performance(pipeline, 'Mika'))
    report.update(nlon_performance(pipeline, 'agreement'))

    with open(OUT_PATH + 'performance.txt', 'w') as fd:
        json.dump(report, fd, indent=2)
    return report, pipeline


if __name__ == "__main__":
    main()
