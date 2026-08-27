import joblib

from src.credibility_features import analyse_text, interpret_features 
from src.combined_assessment import create_assessment 


# Load the saved TF-IDF vectorizer
print("Loading TF-IDF vectorizer...")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
print("TF-IDF vectorizer loaded.")


# Load the saved Logistic Regression model
print("Loading Logistic Regression model...")
model = joblib.load("models/logistic_regression_model.joblib")
print("Logistic Regression model loaded.")


# Get news text from the user
# Get news text from the user

text = input("\nEnter a news article or headline:\n").strip()
# Basic input validation
text = text.strip()

if not text:
    print("\nError: No text was entered.")
    exit()

if len(text) < 20:
    print("\nInput is too short to reliably analyze.")
    print("Please enter a news headline or article with more information.")
    exit()
# Check that the input contains enough alphabetic characters
letter_count = sum(character.isalpha() for character in text)

if letter_count < 10:
    print("\nInput does not contain enough meaningful text to analyze.")
    print("Please enter a news headline or article.")
    exit()

# Preserve the original text for linguistic analysis
original_text = text

# Lowercase a separate copy for the ML model
processed_text = text.lower()

# Transform the processed text using the saved TF-IDF vectorizer
X = tfidf.transform([processed_text])  
# Check how much of the input is represented in the TF-IDF vocabulary

words = processed_text.split() 

known_words = 0

for word in words:
    if word in tfidf.vocabulary_:
        known_words += 1

if known_words == 0:
    print("\nInput does not contain words recognized by the model.")
    print("Please enter a more informative news headline or article.")
    exit()

coverage = known_words / len(words)  

# Make prediction
prediction = model.predict(X)[0]
# Analyse linguistic features
features = analyse_text(original_text) 
indicators = interpret_features(features)  



# Get prediction probabilities
probabilities = model.predict_proba(X)[0]

classes = model.classes_

probability_dict = dict(zip(classes, probabilities))

# Calculate confidence as the highest predicted probability
confidence = max(probabilities)   
# Determine confidence level
if confidence >= 0.80:
    confidence_level = "HIGH"
elif confidence >= 0.60:
    confidence_level = "MEDIUM"
else:
    confidence_level = "LOW" 

assessment = create_assessment(prediction, confidence, indicators["linguistic_risk"])

print("\n===== PREDICTION =====")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence_level} ({confidence:.2%})")
print(f"Vocabulary coverage: {coverage:.2%}") 


print("\nProbabilities:")
for label, probability in probability_dict.items():
    print(f"{label}: {probability:.2%}")

if confidence_level == "LOW":
    print("\nWARNING: The model is not confident in this prediction.")
    print("The result should be treated as uncertain.")
elif confidence_level == "MEDIUM":
    print("\nNOTE: The model has moderate confidence in this prediction.")
else:
    print("\nThe model has high confidence in this prediction.")  

print("\n===== LINGUISTIC ANALYSIS =====")

print(f"Sensational language: {indicators['sensational_language']}")
print(f"Emotional language: {indicators['emotional_language']}")
print(f"Capitalization: {indicators['capitalization']}")
print(f"Punctuation: {indicators['punctuation']}")
print(f"Repeated punctuation: {indicators['repeated_punctuation']}")

print(f"\nLinguistic risk: {indicators['linguistic_risk']}")
print(f"Linguistic risk score: {indicators['linguistic_risk_score']}/10")
print(f"Linguistic risk percentage: {indicators['linguistic_risk_percentage']:.1f}%")  

print("\n===== COMBINED ASSESSMENT =====")
print(assessment["overall_assessment"]) 