# AI-Powered News Credibility Analyser

A machine learning and linguistic analysis tool that helps users assess the credibility of news headlines and articles, built as a YCS project.

**Live demo:** https://news-credibility-analyser.streamlit.app/ 

## The Problem

Misinformation, clickbait, and satire circulate alongside real news, often written in ways that make them hard to tell apart at a glance -- especially for readers under time pressure or without training in media literacy. Sharing unreliable content, even unintentionally, can spread false claims quickly. This tool gives readers a fast, independent second opinion on a headline before they trust or share it, without replacing their own judgement or the need to check trusted sources.

## What This Project Does

This tool takes a news headline or short article and analyses it in two independent ways:

1. **Machine learning prediction** -- a Logistic Regression model trained on TF-IDF text features predicts whether the content is REAL or FAKE, based on patterns learned from a large labelled dataset.
2. **Linguistic analysis** -- a separate rule-based system checks for sensational language, emotional language, excessive capitalisation, and excessive punctuation, all writing-style patterns often associated with unreliable content.

These two signals are combined into a single, clear verdict -- **Likely Reliable**, **Likely Unreliable**, or **Uncertain - Verify** -- along with a plain-language explanation of why. The tool is designed as a decision-support aid, not a replacement for checking trusted sources.

## What Makes This Innovative

Most simple credibility tools present a single confident label. This project is built around a different idea: **two independent, imperfect signals are more trustworthy than one confident-sounding one**, especially when they're allowed to disagree openly. During development, testing surfaced two real, unexpected findings that shaped the final design:

* A satirical headline was confidently misjudged as reliable (85% confidence) before retraining, and even after retraining, the model could only reach 52% confidence on it -- essentially a guess. Rather than forcing a confident label anyway, the system was redesigned to flag any prediction below 65% confidence as "Uncertain - Verify."
* Two genuinely reliable headlines from CNN and The New York Times, describing the *same real event*, received different verdicts purely because of wording differences. This became concrete evidence, gathered through the project's own testing, that the tool measures writing style, not truth -- and is now documented directly in the Limitations section rather than hidden.

The result is a tool that is honest about its own uncertainty instead of always sounding confident, which is a deliberate, tested design choice, not a limitation left unaddressed.

## Setup Instructions

**Requirements:** Python 3.13

1. Clone this repository:

```
   git clone https://github.com/methu-1o1/news-credibility-analyser.git
   cd news-credibility-analyser
   ```

2. Create and activate a virtual environment:

```
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

3. Install the required packages:

```
   pip install -r requirements.txt
   ```

4. Run the web app:

```
   streamlit run app.py
   ```

5. Or run the terminal version instead:

```
   python -m src.predict
   ```

## Methodology

### Dataset

The model was trained on a dataset of real and fake news headlines/articles, combined with additional satire and clickbait examples sourced from public datasets (see Data Sources below). This addition was made after testing revealed the original dataset contained no satire examples, causing the model to misclassify satirical content as reliable.

### Machine Learning Pipeline

1. **Cleaning** -- removed missing values, duplicates, conflicting labels, and very short texts.
2. **Splitting** -- 80% training / 20% testing, stratified to preserve class balance.
3. **Feature extraction** -- TF-IDF vectorisation (unigrams and bigrams, up to 100,000 features).
4. **Model** -- Logistic Regression, trained on the TF-IDF features.
5. **Evaluation** -- accuracy, precision, recall, F1-score, and a confusion matrix.

### Linguistic Analysis

Independently of the ML model, each input is analysed for:

* Sensational language (e.g. "shocking", "bombshell", "exposed")
* Emotional language (e.g. "outrage", "terrifying", "devastating")
* Capitalisation patterns
* Punctuation patterns (excessive "!" or "?", repeated punctuation)

These are converted into a linguistic risk score (Low / Moderate / High).

### Combining the Signals

The ML prediction and linguistic risk score are combined into one of three verdicts:

* **Likely Reliable** -- model predicts REAL, linguistic risk is not high, and confidence is high enough to trust.
* **Likely Unreliable** -- model predicts FAKE and linguistic risk is High (both signals agree).
* **Uncertain - Verify** -- the two signals disagree, or the model's confidence is below 65% (too close to a guess to trust either way).

This design deliberately favours caution: when the model is unsure, or when the linguistic analysis contradicts the model, the tool tells the user to verify independently rather than presenting a confident-sounding but potentially wrong answer.

## Data Sources

* **RealFakeNews Dataset** (fauxNeuz, Hugging Face) -- the primary training dataset of labelled real and fake news headlines/articles.
`huggingface.co/datasets/fauxNeuz/RealFakeNews`
* **News Headlines Dataset For Sarcasm Detection** (Rishabh Misra, Kaggle) -- satirical headlines from The Onion paired with real headlines from HuffPost. Used to add satire examples to the FAKE class.
`kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection`
* **Onion or Not** (Chris Filo Gorgolewski, Kaggle) -- a labelled dataset distinguishing Onion (satirical) headlines from real ones. Used to add further satire examples.
`kaggle.com/datasets/chrisfilo/onion-or-not`
* **Satirical News from The Onion** (Kaggle) -- additional satirical headlines sourced directly from The Onion.
`kaggle.com/datasets/undefinenull/satirical-news-from-the-onion`
* **Clickbait Dataset** (Aman Anand, Kaggle) -- labelled clickbait vs. non-clickbait headlines from sources including BuzzFeed, Upworthy, and legitimate outlets like The New York Times. Used to add clickbait examples to the FAKE class.
`kaggle.com/datasets/amananandrai/clickbait-dataset`

A random sample of 3,000 rows was taken from each of the three additional datasets and merged into the original training data as FAKE-labelled examples, after which duplicates and conflicting labels were removed during cleaning.

## Future Potential and Sustainability

This project is intentionally built with low-cost, freely available components (public datasets, open-source Python libraries, free hosting on Streamlit Community Cloud), so it can continue to be improved and run without ongoing cost. Realistic directions for future growth include:

* **Better satire handling** -- sourcing a larger and more diverse set of labelled satire examples, or exploring a more expressive model architecture, to move satire detection from "uncertain" to "confidently correct."
* **Browser extension or API** -- packaging the credibility check as a right-click browser extension or a simple API, so it could be used directly while reading news elsewhere, rather than requiring a user to copy-paste text into a separate app.
* **Source-level credibility signals** -- extending the analysis to also consider the publishing domain's general track record, alongside the text itself.
* **Multilingual support** -- the current linguistic keyword lists and training data are English-only; expanding to Sinhala and Tamil headlines would make the tool useful to a much wider local audience.
* **Community feedback loop** -- letting users flag verdicts they disagree with, to build a growing, real-world-corrected dataset over time.

The tool's core design -- combining an ML signal with an independent linguistic signal and being explicit about disagreement -- is not tied to news credibility specifically, and the same approach could extend to other domains where a single confident automated label would be risky to trust blindly.

## Limitations

* **Satire detection is imperfect.** Even after retraining with satire and clickbait examples, deadpan satirical headlines (e.g. from The Onion) are often predicted with low confidence rather than being confidently and correctly flagged as unreliable. For example, the headline "HHS Replaces Childhood Vaccination Schedule With List Of Recommended Animal Bites" was predicted FAKE at only 52.48% confidence -- essentially a guess. This is because satire deliberately mimics the calm, formal style of real journalism, which the linguistic analysis (designed to catch sensational language) does not detect, and which a TF-IDF model can only learn if it has seen enough similar examples during training.
* **The ML model reflects patterns in its training data, not real-world truth.** A Logistic Regression model trained on TF-IDF features learns statistical word associations, not factual accuracy. It cannot verify claims, check sources, or reason about plausibility -- it can only recognise writing style patterns similar to what it was trained on.
* **Low-confidence predictions are flagged, not hidden.** To address the above, the system treats any ML prediction below 65% confidence as "Uncertain - Verify," regardless of the label. This makes the tool more honest, but also means it will sometimes decline to give a confident answer even when a human might correctly judge the content as reliable or unreliable at a glance.
* **The linguistic analysis uses a fixed keyword list.** Sensational and emotional word detection relies on predefined word lists, which will miss synonyms, sarcasm, or non-English text, and may occasionally flag legitimate journalistic language (e.g. "the disaster left ten dead" is a normal real-news sentence but contains an "emotional" keyword).
* **The model can disagree with itself on the same real event.** Two credible headlines describing the same real story were tested: a CNN headline, "Trump says U.S. reaches deal with Venezuela to control 65 billion barrels of country's oil reserves," and a New York Times headline on the same event, "Trump Says U.S. Has Deal for Control of a Large Share of Venezuela's Oil." The CNN headline was assessed as "Likely Reliable" (74.58% confidence), while the NYT headline -- reporting the identical event -- was assessed as "Uncertain - Verify" (58.14% confidence). Both headlines are genuinely reliable; the difference in outcome came from wording alone (e.g. a specific number vs. a vaguer phrase), not from any actual difference in credibility. This is a clear demonstration that the model scores writing style and word patterns, not the truth or reliability of the underlying story.
* **This tool is a decision-support aid, not a fact-checker.** It cannot verify whether specific claims within an article are true. Users should always cross-check important information with trusted, independent sources.

## Responsible AI Discussion

This project is designed with the understanding that automated credibility tools carry real risks if misused or over-trusted:

* **False confidence is dangerous.** A tool that confidently mislabels fake content as real (or vice versa) can do more harm than having no tool at all, by giving users a false sense of certainty. This is why the system was deliberately redesigned during development to surface uncertainty rather than hide it -- favouring "Uncertain - Verify" over a confident but potentially wrong answer whenever the model's confidence is low or its signals disagree.
* **No automated system should be the final word on truth.** This tool assesses writing patterns and statistical associations, not factual accuracy. It is not capable of fact-checking specific claims, and it says so directly in the app's interface.
* **Bias in training data becomes bias in the model.** The datasets used here reflect the sources they were collected from (e.g. specific news outlets, specific satire publications), so the model's judgement is shaped by those sources' conventions and may not generalise perfectly to all writing styles, topics, or contexts.
* **Transparency matters.** Rather than presenting a single opaque "real/fake" label, this tool shows its reasoning -- the ML confidence, the specific linguistic indicators detected, and why the two were combined the way they were -- so users can understand and question the result rather than simply accept it.

The overall goal of this project is to support more informed reading habits, not to replace critical thinking or independent verification.

## Project Structure

```
news-credibility-analyser/
|-- app.py                          (Streamlit web app)
|-- requirements.txt                (Python dependencies)
|-- data/
|   |-- raw/                        (Original and additional source datasets - not committed, see .gitignore)
|   |-- processed/                  (Cleaned, split, and analysis output files)
|-- models/
|   |-- logistic_regression_model.joblib
|   |-- tfidf_vectorizer.joblib
|   |-- confusion_matrix.png
|-- src/
|   |-- clean_dataset.py            (Data cleaning)
|   |-- split_dataset.py            (Train/test splitting)
|   |-- train_model.py              (Model training)
|   |-- evaluate_model.py           (Model evaluation)
|   |-- credibility_features.py     (Linguistic analysis)
|   |-- combined_assessment.py      (Combines ML + linguistic signals into a verdict)
|   |-- merge_satire_clickbait.py   (Merges satire/clickbait datasets into training data)
|   |-- predict.py                  (Terminal-based prediction tool)
|   |-- inspect_dataset.py
|   |-- error_analysis.py
|   |-- confidence_analysis.py
|   |-- plot_confidence.py
|-- tests/
|   |-- test_predict.py             (Automated tests)
```

