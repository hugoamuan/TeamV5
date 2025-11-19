# services/match_service.py
import re

def tokenize(text: str):
    """
    Splits text into clean alphanumeric+symbol tokens.
    Ensures skills like:
    - java
    - c++
    - c#
    - sql
    - go
    are matched properly without false positives.
    """
    return re.findall(r"[A-Za-z0-9\+\#]+", text.lower())

def job_matches_skills(description: str, skills: list[str], threshold: float = 0.25):
    """
    Matches user skills to job description using exact token matching.

    Example:
    - 'java' will NOT match 'javascript'
    - 'rust' will NOT match 'trusted'
    - 'go' will NOT match 'google'
    - 'c' will NOT match 'react'

    Matching is reliable and avoids substring false positives.
    """

    if not description:
        return False

    # Tokenize the job description once
    tokens = tokenize(description)
    token_set = set(tokens)

    matched = 0

    for skill in skills:
        skill_clean = skill.lower().strip()
        if not skill_clean:
            continue

        # Token-based exact comparison
        if skill_clean in token_set:
            matched += 1

    ratio = matched / len(skills) if skills else 0
    return ratio >= threshold

