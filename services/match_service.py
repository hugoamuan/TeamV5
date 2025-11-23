# services/match_service.py
# Job to skill matching service

import re

# Converts text into lowercase tokens to avoid bad matches such as "Go" in "Google" or "c" in "react"
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

# returns true if enough skills appear in the job description
# Threshold 0.25 = if the user has 4 skills needs 1 match (25%)
#                  ............... 8 skills ..... 2 matches
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

    # calculate match ratio
    ratio = matched / len(skills) if skills else 0
    return ratio >= threshold

