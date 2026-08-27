def create_assessment(prediction, confidence, linguistic_risk):
    """
    Combine the machine-learning prediction and linguistic analysis
    into a transparent human-readable assessment.

    Linguistic risk is a separate writing-style signal and does not
    prove that an article is fake or true.
    """

    assessment = {}

    assessment["prediction"] = prediction
    assessment["ml_confidence"] = confidence
    assessment["linguistic_risk"] = linguistic_risk

    if prediction == "FAKE":
        if linguistic_risk == "High":
            assessment["overall_assessment"] = (
                "The model predicts FAKE with high linguistic risk."
            )
        elif linguistic_risk == "Moderate":
            assessment["overall_assessment"] = (
                "The model predicts FAKE with moderate linguistic risk."
            )
        else:
            assessment["overall_assessment"] = (
                "The model predicts FAKE, but the linguistic risk is low."
            )

    elif prediction == "REAL":
        if linguistic_risk == "High":
            assessment["overall_assessment"] = (
                "The model predicts REAL, but the text contains "
                "high-risk linguistic patterns."
            )
        elif linguistic_risk == "Moderate":
            assessment["overall_assessment"] = (
                "The model predicts REAL, with some linguistic risk indicators."
            )
        else:
            assessment["overall_assessment"] = (
                "The model predicts REAL and the linguistic risk is low."
            )

    else:
        assessment["overall_assessment"] = (
            "The model produced an unexpected prediction."
        )

    return assessment 