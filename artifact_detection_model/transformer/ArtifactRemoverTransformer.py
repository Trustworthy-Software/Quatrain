import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from artifact_detection_model.transformer.DoNotReplaceArtifacts import DoNotReplaceArtifacts
from artifact_detection_model.transformer.ReplaceButKeepExceptionNames import ReplaceButKeepExceptionNames
from artifact_detection_model.transformer.SimpleReplace import SimpleReplace
from artifact_detection_model.utils.paths import PROJECT_ROOT

classifier = joblib.load(PROJECT_ROOT + 'artifact_detection_model/out/artifact_detection.joblib')

SIMPLE = 'simple'
KEEP_EXCEPTION_NAMES = 'keep_exception_names'
DO_NOT_REPLACE = 'no_replacements'

replacement_strategies = {SIMPLE: SimpleReplace(),
                          KEEP_EXCEPTION_NAMES: ReplaceButKeepExceptionNames(),
                          DO_NOT_REPLACE: DoNotReplaceArtifacts()}


class ArtifactRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, replacement_strategy=SIMPLE):
        self.replacement_strategy = replacement_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.replacement_strategy == DO_NOT_REPLACE:
            return X
        return [self.predict_and_remove(i) for i in X]

    def predict_and_remove(self, issue):
        replacement_strategy = replacement_strategies[self.replacement_strategy]

        prediction = classifier.predict(issue.splitlines())
        text_indices = [i for i, e in enumerate(prediction) if e == 1]
        cleaned_issue = []
        for i in range(0, len(issue.splitlines())):
            if i in text_indices:
                cleaned_issue.append(issue.splitlines()[i])
            else:
                replacement = replacement_strategy.get_replacement(issue.splitlines()[i])
                if replacement.strip():
                    cleaned_issue.append(replacement)
        return '\n'.join(cleaned_issue)

    def transform2(self, X):
        if self.replacement_strategy == DO_NOT_REPLACE:
            return X
        return [self.predict_and_remove2(i) for i in X]

    def predict_and_remove2(self, issue):
        prediction = classifier.predict(issue.splitlines())
        text_indices = [i for i, e in enumerate(prediction) if e == 1]
        cleaned_issue = []
        codes = []
        for i in range(0, len(issue.splitlines())):
            if i in text_indices:
                cleaned_issue.append(issue.splitlines()[i])
            else:
                codes.append(issue.splitlines()[i])
        return '\n'.join(cleaned_issue+codes)
