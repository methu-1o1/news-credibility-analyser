import re


SENSATIONAL_WORDS = {
    "shocking",
    "shocked",
    "unbelievable",
    "outrageous",
    "bombshell",
    "explosive",
    "explodes",
    "stunning",
    "stunned",
    "incredible",
    "secret",
    "exposed",
    "breaking",
    "urgent",
    "scandal",
    "scandalous",
    "miracle",
    "terrifying",
    "horrifying",
    "destroyed",
    "slammed",
    "blasted",
}


EMOTIONAL_WORDS = {
    "angry",
    "anger",
    "fear",
    "afraid",
    "terrified",
    "terrifying",
    "hate",
    "hated",
    "horrible",
    "horrific",
    "horrifying", 
    "disgusting",
    "disaster",
    "tragic",
    "devastating",
    "devastated",
    "furious",
    "rage",
    "panic",
    "panicked",
    "danger",
    "dangerous",
}


def analyse_text(text):
    """
    Analyse linguistic characteristics of a news headline or article.
    """

    text = text.strip()

    words = text.split()

    word_count = len(words)

    character_count = len(text)

    # Normalise words for dictionary matching
    normalised_words = [
        re.sub(r"[^a-zA-Z]", "", word.lower())
        for word in words
    ]

    # Sensational-language indicators
    sensational_matches = [
        word
        for word in normalised_words
        if word in SENSATIONAL_WORDS
    ]

    sensational_count = len(sensational_matches)

    if word_count > 0:
      sensational_ratio = sensational_count / word_count
    else:  
      sensational_ratio = 0  

    # Emotional-language indicators
    emotional_matches = [
        word
        for word in normalised_words
        if word in EMOTIONAL_WORDS
    ]

    emotional_count = len(emotional_matches)

    if word_count > 0:
       emotional_ratio = emotional_count / word_count
    else:
       emotional_ratio = 0   

    # Capitalisation analysis
    uppercase_count = sum(
        1 for character in text
        if character.isupper()
    )

    letter_count = sum(
        1 for character in text
        if character.isalpha()
    )

    if letter_count > 0:
        uppercase_ratio = uppercase_count / letter_count
    else:
        uppercase_ratio = 0

          # Detect words written entirely in uppercase
    uppercase_word_count = sum(
        1
        for word in words
        if any(character.isalpha() for character in word)
        and word == word.upper()
    )   
     

    # Punctuation analysis
    exclamation_count = text.count("!")
    question_count = text.count("?")

    repeated_punctuation_count = len(
        re.findall(r"[!?]{2,}", text)
    )

    # URL detection
    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", text)
    )

    # Average word length
    if words:
        average_word_length = sum(
            len(word.strip(".,!?;:\"'()[]{}"))
            for word in words
        ) / len(words)
    else:
        average_word_length = 0

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sensational_word_count": sensational_count,
        "sensational_words": sensational_matches,
        "emotional_word_count": emotional_count,
        "emotional_words": emotional_matches,
        "uppercase_ratio": uppercase_ratio,
        "uppercase_word_count": uppercase_word_count, 
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "repeated_punctuation_count": repeated_punctuation_count,
        "url_count": url_count,
        "average_word_length": average_word_length,
        "sensational_ratio": sensational_ratio,
        "emotional_ratio": emotional_ratio,  
    }

def interpret_features(features):
    """
    Convert raw linguistic features into human-readable indicators.
    These indicators describe writing patterns and do not prove whether
    a piece of news is true or false.
    """

    indicators = {}

    # Sensational language
    sensational_ratio = features["sensational_ratio"]

    if sensational_ratio == 0:
       indicators["sensational_language"] = "Low"
    elif sensational_ratio <= 0.10:
       indicators["sensational_language"] = "Moderate"
    else:
       indicators["sensational_language"] = "High"  

    # Emotional language
    emotional_ratio = features["emotional_ratio"]

    if emotional_ratio == 0:
       indicators["emotional_language"] = "Low"
    elif emotional_ratio <= 0.10:
       indicators["emotional_language"] = "Moderate"
    else:
       indicators["emotional_language"] = "High"  
      

        # Capitalization

    uppercase_word_count = features["uppercase_word_count"]

    if uppercase_word_count == 0:
        indicators["capitalization"] = "Normal"
    elif uppercase_word_count == 1:
        indicators["capitalization"] = "Elevated"
    else:
        indicators["capitalization"] = "High"

    # Punctuation
    punctuation_count = (
        features["exclamation_count"]
        + features["question_count"]
    )

    if punctuation_count == 0:
        indicators["punctuation"] = "Normal"
    elif punctuation_count <= 2:
        indicators["punctuation"] = "Elevated"
    else:
        indicators["punctuation"] = "High"

    # Repeated punctuation
        # Repeated punctuation

    repeated_count = features["repeated_punctuation_count"]

    if repeated_count == 0:
        indicators["repeated_punctuation"] = "None"
    elif repeated_count == 1:
        indicators["repeated_punctuation"] = "Present"
    else:
        indicators["repeated_punctuation"] = "Frequent"

    # Overall linguistic risk score
    risk_score = 0

    if indicators["sensational_language"] == "Moderate":
        risk_score += 1
    elif indicators["sensational_language"] == "High":
        risk_score += 2

    if indicators["emotional_language"] == "Moderate":
        risk_score += 1
    elif indicators["emotional_language"] == "High":
        risk_score += 2

    if indicators["capitalization"] == "Elevated":
        risk_score += 1
    elif indicators["capitalization"] == "High":
        risk_score += 2

    if indicators["punctuation"] == "Elevated":
        risk_score += 1
    elif indicators["punctuation"] == "High":
        risk_score += 2

    if indicators["repeated_punctuation"] == "Present":
        risk_score += 1
    elif indicators["repeated_punctuation"] == "Frequent":
        risk_score += 2

    # Convert score into a human-readable assessment
    if risk_score <= 2:
        indicators["linguistic_risk"] = "Low"
    elif risk_score <= 5:
        indicators["linguistic_risk"] = "Moderate"
    else:
        indicators["linguistic_risk"] = "High"

    indicators["linguistic_risk_score"] = risk_score
    indicators["linguistic_risk_percentage"] = (risk_score / 10) * 100 

    return indicators   
if __name__ == "__main__":

    sample_text = "Read the full report at https://example.com/news/article" 

    features = analyse_text(sample_text)

    print("===== CREDIBILITY FEATURES =====")

    for name, value in features.items():
        print(f"{name}: {value}")

    indicators = interpret_features(features)

    print("\n===== LINGUISTIC INDICATORS =====")

    for name, value in indicators.items():
        print(f"{name}: {value}")    