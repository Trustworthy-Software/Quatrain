from artifact_detection_model.transformer.ReplaceStrategy import ReplaceStrategy


class DoNotReplaceArtifacts(ReplaceStrategy):
    def get_replacement(self, text):
        return text

    def perform_replace(self, rex, text):
        return text
