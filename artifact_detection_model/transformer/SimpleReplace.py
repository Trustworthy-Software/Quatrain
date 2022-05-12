from artifact_detection_model.transformer.ReplaceStrategy import ReplaceStrategy


class SimpleReplace(ReplaceStrategy):
    def get_replacement(self, text):
        return ' '

    def perform_replace(self, rex, text):
        return rex.sub(' ', text)
