import time
import timeit

import joblib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from artifact_detection_model.SpecialCharacterToWords import SpecialCharacterToWords
from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.utils.Logger import Logger

log = Logger()


def run_ml_artifact_training(df_train, df_test, clf, model_output_path=''):
    data_train = df_train.copy().pop('doc').values
    target_train = df_train.copy().pop('target').values
    data_validation = df_test.copy().pop('doc').values
    target_validation = df_test.copy().pop('target').values

    pipeline = Pipeline([
        ('charrep', SpecialCharacterToWords()),
        ('vect', CountVectorizer()),
        ('clf', clf)])

    parameters = {
        'charrep__repl_all_caps': False,
        'vect__ngram_range': (1, 3),
        'vect__stop_words': None,
        'vect__lowercase': False,
    }

    log.s("train_samples: %d" % len(data_train))
    log.s("test_samples: %d" % len(data_validation))

    perf_start = time.perf_counter()

    pipeline.set_params(**parameters)
    pipeline.fit(data_train, target_train)

    perf_train_runtime = time.perf_counter() - perf_start

    num_runs_timeit = 10
    timeit_runtime = timeit.timeit(stmt='pipeline.predict(data_validation)', number=num_runs_timeit, globals={'pipeline': pipeline, 'data_validation': data_validation}) / num_runs_timeit

    perf_start = time.perf_counter()
    y_predicted = pipeline.predict(data_validation)
    perf_runtime = time.perf_counter() - perf_start

    log.s(metrics.classification_report(target_validation, y_predicted, target_names=TARGET_NAMES))
    log.s('\nroc_auc=' + str(roc_auc_score(target_validation, y_predicted)))
    log.s('\nperf_runtime=' + str(perf_runtime))

    confusion_matrix(target_validation, y_predicted)

    performance_report = {'train_samples': len(data_train),
                          'test_samples': len(data_validation),
                          'params': str(parameters),
                          'classification_report': str(metrics.classification_report(target_validation, y_predicted, target_names=TARGET_NAMES)),
                          'macro_f1': f1_score(target_validation, y_predicted, average='macro'),
                          'roc-auc': roc_auc_score(target_validation, y_predicted),
                          'perf_runtime': perf_runtime,
                          'timeit_runtime': timeit_runtime,
                          'perf_train_runtime': perf_train_runtime}

    if model_output_path:
        with open(model_output_path + 'out.txt', 'w') as fd:
            fd.write("train_samples:  " + str(len(data_train)) + '\n')
            fd.write("test_samples:  " + str(len(data_validation)) + '\n')
            # fd.write('score: ' + str(grid_search.best_score_) + '\n')
            fd.write('params: ' + str(parameters) + '\n')
            # fd.write('estimator: ' + str(grid_search.best_estimator_) + '\n')
            fd.write('\n')
            fd.write(str(metrics.classification_report(target_validation, y_predicted, target_names=TARGET_NAMES)))
            fd.write('\nroc_auc=' + str(roc_auc_score(target_validation, y_predicted)) + '\n')
            fd.write('\nperf_runtime=' + str(perf_runtime) + '\n')
            fd.write(str(confusion_matrix(target_validation, y_predicted)))

        disp = plot_confusion_matrix(pipeline, data_validation, target_validation,
                                     display_labels=TARGET_NAMES,
                                     normalize=None)
        print(disp.confusion_matrix)

        plt.savefig(model_output_path + 'confusionmatrix.png')

        joblib.dump(pipeline, model_output_path + 'artifact_detection.joblib')

        wrongly_identified_as_artifact = []
        wrongly_identified_as_text = []
        for index in range(0, len(data_validation)):
            if target_validation[index] == y_predicted[index]:
                pass
            elif target_validation[index] == TARGET_NAMES['artifact'] and y_predicted[index] == TARGET_NAMES['text']:
                wrongly_identified_as_text.append(data_validation[index])
            else:
                wrongly_identified_as_artifact.append(data_validation[index])

        with open(model_output_path + 'wrongly_identified_as_artifact.json', 'w') as fd:
            fd.write('\n\n'.join(wrongly_identified_as_artifact))
        with open(model_output_path + 'wrongly_identified_as_text.json', 'w') as fd:
            fd.write('\n\n'.join(wrongly_identified_as_text))

    return performance_report, pipeline