import re


class CodeFeeder:
    def __init__(self):
        self.patterns = [
            r"\bimport\s+(\w+)",
            r"\bfrom\s+(\w+)\s+import",
            r"\bclass\s+(\w+)",
            r"\bdef\s+(\w+)",
            r"\btry:",
            r"\bexcept\s+(\w+):",
            r"\bfinally:",
            r"\bif\s+(\w+):",
            r"\belif\s+(\w+):",
            r"\belse:",
            r"\bwhile\s+(\w+):",
            r"\bfor\s+(\w+)\s+in",
            r"\breturn\s+(\w+)",
            r"\bglobal\s+(\w+)",
            r"\bassert\s+(\w+)",
            r"\bwith\s+(\w+):"
        ]

    def extract_code(self, text):
        code = []
        for line in text.split("\n"):
            line = line.strip()
            for pattern in self.patterns:
                match = re.search(pattern, line)
                if match:
                    code.append(match.group(1))
                    break
        return code
