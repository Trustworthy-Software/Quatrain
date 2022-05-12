from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from artifact_detection_model.transformer.ArtifactRemoverTransformer import ArtifactRemoverTransformer, DO_NOT_REPLACE, \
    SIMPLE, KEEP_EXCEPTION_NAMES


def run_classifier(X_train, X_test, y_train, y_test, target_names):
    pipeline = Pipeline([
                         ('artifacts_rem', ArtifactRemoverTransformer()),
                         ('vect', CountVectorizer(stop_words='english', ngram_range=(1, 1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', LinearSVC(random_state=42, C=1))])

    parameters = {
        'artifacts_rem__replacement_strategy': [DO_NOT_REPLACE, SIMPLE, KEEP_EXCEPTION_NAMES],
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=10, scoring='f1_macro')
    grid_search.fit(X_train, y_train)
    y_predicted = grid_search.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted, target_names=target_names))
