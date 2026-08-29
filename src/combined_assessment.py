def create_assessment(prediction, confidence, linguistic_risk, confidence_threshold=0.65):
    """
    Combine the machine-learning prediction and linguistic analysis
    into a transparent, human-readable assessment.

    Linguistic risk is a separate writing-style signal and does not
    prove that an article is fake or true. When the two signals
    disagree, or when the model's confidence is low, the system
    favours caution over a confident-sounding label.
    """

    assessment = {}

    assessment["prediction"] = prediction
    assessment["ml_confidence"] = confidence
    assessment["linguistic_risk"] = linguistic_risk

    ml_says_fake = (prediction == "FAKE")
    linguistic_flags_risk = (linguistic_risk == "High")
    low_confidence = (confidence < confidence_threshold)

    if low_confidence:
        assessment["final_verdict"] = "Uncertain — Verify"
        assessment["overall_assessment"] = (
            f"The model's confidence in this prediction is low "
            f"({confidence:.1%}), close to a guess. This result should "
            f"not be trusted at face value — verify independently."
        )

    elif ml_says_fake and linguistic_flags_risk:
        assessment["final_verdict"] = "Likely Unreliable"
        assessment["overall_assessment"] = (
            "Both the model and the writing style suggest this content "
            "is likely unreliable."
        )

    elif ml_says_fake and not linguistic_flags_risk:
        assessment["final_verdict"] = "Uncertain — Verify"
        assessment["overall_assessment"] = (
            "The model flags this as potentially unreliable, though the "
            "writing style itself doesn't show strong warning signs. "
            "Treat this result as uncertain and verify independently."
        )

    elif not ml_says_fake and linguistic_flags_risk:
        assessment["final_verdict"] = "Uncertain — Verify"
        assessment["overall_assessment"] = (
            "The model predicts this is real, but the text shows "
            "high-risk sensational or emotional writing patterns. "
            "This disagreement means the result should not be trusted "
            "at face value — verify independently."
        )

    else:
        assessment["final_verdict"] = "Likely Reliable"
        assessment["overall_assessment"] = (
            "Both the model and the writing style suggest this content "
            "is likely reliable. This is still not a guarantee of truth."
        )

    return assessment 