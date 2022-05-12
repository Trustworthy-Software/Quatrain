import re

from artifact_detection_model.transformer.ReplaceStrategy import ReplaceStrategy

keeprex = [re.compile(r"(?:(?:[A-Z][a-z0-9]*)+(?:Exception|Error))")]


class ReplaceButKeepExceptionNames(ReplaceStrategy):
    def get_replacement(self, text):
        reduced = []
        for rex in keeprex:
            reduced.extend(rex.findall(text))
        return ' ' + ' '.join(reduced) + ' '

    def perform_replace(self, rex, text):
        for match in rex.findall(text):
            text = text.replace(match, self.get_replacement(match))
        return text