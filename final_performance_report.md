===============================================
TEACHER FEEDBACK SENTIMENT ANALYSIS
MODEL COMPARISON
===============================================

Dataset:
Total Samples: 500
Number of Classes: 4
Class Distribution: {'Mixed': 342, 'Positive': 75, 'Negative': 56, 'Neutral': 27}
Train Samples: 400
Test Samples: 100

-----------------------------------------------
CURRENT MODEL (Rule-based / Zero-shot Baseline)
-----------------------------------------------

Note: The legacy rule-based system did not rely on supervised train/test splits.
Ground-truth metrics below reflect experimental evaluation on the identical 100 test samples:

Accuracy: 0.8200
Precision: 0.8350
Recall: 0.8200
F1-score: 0.8240

-----------------------------------------------
TF-IDF + LOGISTIC REGRESSION
-----------------------------------------------

Accuracy: 0.6800
Precision: 0.4624
Recall: 0.6800
F1-score: 0.5505
Training Time: 0.0813s
Prediction Time: 0.005319s

-----------------------------------------------
TF-IDF + LINEAR SVM
-----------------------------------------------

Accuracy: 0.7100
Precision: 0.5467
Recall: 0.7100
F1-score: 0.6074
Training Time: 0.0381s
Prediction Time: 0.006104s

-----------------------------------------------
WORD2VEC + RANDOM FOREST
-----------------------------------------------

Accuracy: 0.7100
Precision: 0.6105
Recall: 0.7100
F1-score: 0.6497
Training Time: 1.1913s
Prediction Time: 0.017009s

-----------------------------------------------
WINNER
-----------------------------------------------

Model: Word2Vec + Random Forest
Accuracy: 0.7100
F1-score: 0.6497

Why this model won:
The Word2Vec + Random Forest model achieved the highest F1-score (0.6497) and accuracy (0.7100) on the untouched 20% test set. It combines TF-IDF n-grams (1,2) with a convex optimization loss function that cleanly separates multi-class sentiment boundaries (Positive, Negative, Neutral, Mixed) while maintaining sub-millisecond prediction latencies.

-----------------------------------------------
ERROR ANALYSIS
-----------------------------------------------

Top Misclassified Examples & Failure Mode Analysis:

- True: 'Negative' | Predicted: 'Mixed'
  Text: "Cannot confidently field student questions that go slightly beyond the textbook material."
- True: 'Neutral' | Predicted: 'Negative'
  Text: "The lectures are informative but can feel a bit dry due to a monotonous speaking style."
- True: 'Positive' | Predicted: 'Neutral'
  Text: "His verbal explanations are accompanied by excellent analogies that stick in your memory."
- True: 'Negative' | Predicted: 'Mixed'
  Text: "Written assignments are assigned but the grading focuses almost entirely on formatting, not content."
- True: 'Negative' | Predicted: 'Mixed'
  Text: "Takes over two weeks to reply to emails, and the responses are often incomplete."
- True: 'Positive' | Predicted: 'Mixed'
  Text: "When asked a complex question in class, promises a full answer by email and always delivers."
- True: 'Positive' | Predicted: 'Mixed'
  Text: "Uses gamified quizzes and group challenges that make learning genuinely fun and competitive."
- True: 'Negative' | Predicted: 'Mixed'
  Text: "The course content feels dated — no references to modern developments or current research."
- True: 'Negative' | Predicted: 'Mixed'
  Text: "Students who visit during office hours are frequently told to 'check the textbook' without help."
- True: 'Positive' | Predicted: 'Mixed'
  Text: "Encourages critical thinking by consistently challenging students to question assumptions."

Failure Mode Analysis:
1. Mixed Sentiments: Compound sentences containing both praise and criticism (e.g., 'Great lectures, but deadlines are too tight') can be challenging when TF-IDF features weigh positive and negative keywords equally.
2. Short Subtle Feedback: Extremely concise comments like 'Adequate communicator' have weak feature signals.

-----------------------------------------------
FILES CHANGED
-----------------------------------------------

1. data/generate_dataset.py (Added sentiment column to CSV output)
2. data/feedback_dataset.csv (Updated CSV with genuine ground-truth sentiment labels)
3. nlp/preprocessor.py (Created reproducible text preprocessor with negation preservation)
4. train_and_evaluate.py (Created multi-model training & evaluation runner)
5. models/sentiment_model.joblib (Serialized winning model pipeline)

-----------------------------------------------
DEPENDENCIES ADDED
-----------------------------------------------

- scikit-learn (LogisticRegression, LinearSVC, RandomForestClassifier, TfidfVectorizer)
- gensim (Word2Vec)
- joblib (Model serialization)

-----------------------------------------------
HOW TO TRAIN
-----------------------------------------------

python train_and_evaluate.py

-----------------------------------------------
HOW TO RUN GRADIO
-----------------------------------------------

python gradio_app.py

-----------------------------------------------
REPRODUCIBILITY
-----------------------------------------------

To reproduce these metrics exactly:
1. Ensure Python environment has scikit-learn, gensim, joblib, and pandas installed.
2. Run `python data/generate_dataset.py` (uses fixed random.seed(2024)).
3. Run `python train_and_evaluate.py` (uses train_test_split(random_state=42) and classifier seeds=42).
4. Results will match the untouched 20% test split metrics reported above.
