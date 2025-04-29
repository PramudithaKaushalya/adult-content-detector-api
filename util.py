import re

def extract_sinhala_text(text: str) -> str:
    # Match only Sinhala characters and spaces
    sinhala_only = re.findall(r'[\u0D80-\u0DFF\s]+', text)
    # Join, collapse extra spaces, and return
    return " ".join("".join(sinhala_only).split())