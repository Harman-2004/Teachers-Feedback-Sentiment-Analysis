"""
tests/test_ml_pipeline.py
--------------------------
Unit tests for the ML pipeline, text preprocessor, model serialization,
and Gradio prediction endpoints.
"""

import os
import pytest
from nlp.preprocessor import preprocess_text, preprocess_documents
from nlp.sentiment import load_ml_model, analyze_sentiment_ml
from gradio_app import predict_sentiment_gradio, predict_batch_gradio

os.environ["FORCE_RULE_BASED"] = "1"


def test_preprocessor_negation_preservation():
    raw = "The explanations are NOT satisfied and never helpful!"
    cleaned = preprocess_text(raw)
    assert "not" in cleaned
    assert "never" in cleaned
    assert "satisfied" in cleaned


def test_preprocessor_documents():
    docs = ["Great teacher!", "Poor organization..."]
    cleaned_docs = preprocess_documents(docs)
    assert len(cleaned_docs) == 2
    assert "great" in cleaned_docs[0]


def test_ml_model_loading():
    model = load_ml_model()
    assert model is not None
    assert hasattr(model, "predict")


def test_ml_sentiment_positive():
    text = "Creates an incredibly interactive learning environment where every student feels heard."
    results = analyze_sentiment_ml([text])
    assert len(results) == 1
    assert results[0]["label"] in ["Positive", "Mixed"]


def test_ml_sentiment_negative():
    text = "Feedback on assignments is generic, copy-pasted, and returns no useful guidance."
    results = analyze_sentiment_ml([text])
    assert len(results) == 1
    assert results[0]["label"] in ["Negative", "Mixed"]


def test_ml_sentiment_empty_input():
    results = analyze_sentiment_ml([""])
    assert len(results) == 1
    assert results[0]["label"] in ["Neutral", "Mixed", "Positive", "Negative"]


def test_gradio_single_prediction():
    text = "Explains every topic with exceptional clarity."
    label, conf, pol, details = predict_sentiment_gradio(text)
    assert label in ["Positive", "Mixed", "Neutral", "Negative"]
    assert isinstance(conf, float)
    assert "10.0" in pol


def test_gradio_empty_input():
    label, conf, pol, details = predict_sentiment_gradio("")
    assert label == "Empty Input"
    assert conf == 0.0


def test_gradio_batch_prediction():
    batch = "Great teacher!\nUnresponsive to emails."
    df_res = predict_batch_gradio(batch)
    assert len(df_res) == 2
    assert "Feedback Text" in df_res.columns
